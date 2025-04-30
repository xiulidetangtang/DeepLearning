from abc import abstractmethod
import numpy as np


class Layer():
    def __init__(self) -> None:
        self.optimizable = True

    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):


    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False,
                 weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(scale=1 / np.sqrt(in_dim), size=(in_dim, out_dim))
        self.b = initialize_method(scale=0.01, size=(1, out_dim))
        self.grads = {'W': None, 'b': None}
        self.input = None

        self.params = {'W': self.W, 'b': self.b}

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):

        self.input = X
        out = np.dot(X, self.W) + self.b
        return out

    def backward(self, grad: np.ndarray):


        dW = np.dot(self.input.T, grad)
        db = np.sum(grad, axis=0, keepdims=True)

        if self.weight_decay:
            dW += self.weight_decay_lambda * self.W

        self.grads['W'] = dW
        self.grads['b'] = db

        dx = np.dot(grad, self.W.T)

        return dx

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}


class conv2D(Layer):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal,
                 weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        scale = 1.0 / np.sqrt(in_channels * kernel_size[0] * kernel_size[1])
        self.W = initialize_method(scale=scale, size=(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.b = initialize_method(scale=0.01, size=(out_channels,))

        self.stride = stride
        self.padding = padding
        self.grads = {'W': None, 'b': None}
        self.params = {'W': self.W, 'b': self.b}
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

        self.input = None
        self.input_padded = None

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def _pad(self, X):
        if self.padding == 0:
            return X

        batch_size, channels, height, width = X.shape
        padded = np.zeros((batch_size, channels, height + 2 * self.padding, width + 2 * self.padding))
        padded[:, :, self.padding:self.padding + height, self.padding:self.padding + width] = X

        return padded

    def _im2col(self, X, kernel_height, kernel_width, stride):

        batch_size, channels, height, width = X.shape
        out_height = (height - kernel_height) // stride + 1
        out_width = (width - kernel_width) // stride + 1

        im_col = np.zeros((batch_size, channels * kernel_height * kernel_width, out_height * out_width))

        for idx_batch in range(batch_size):
            col_idx = 0
            for i in range(0, height - kernel_height + 1, stride):
                for j in range(0, width - kernel_width + 1, stride):
                    im_col[idx_batch, :, col_idx] = X[idx_batch, :, i:i + kernel_height, j:j + kernel_width].reshape(-1)
                    col_idx += 1

        return im_col

    def forward(self, X):

        self.input = X
        batch_size, channels, height, width = X.shape

        X_padded = self._pad(X)
        self.input_padded = X_padded

        out_channels, _, kernel_h, kernel_w = self.W.shape
        out_h = (X_padded.shape[2] - kernel_h) // self.stride + 1
        out_w = (X_padded.shape[3] - kernel_w) // self.stride + 1

        output = np.zeros((batch_size, out_channels, out_h, out_w))

        for b in range(batch_size):
            for c_out in range(out_channels):
                for h_out in range(out_h):
                    h_in = h_out * self.stride
                    for w_out in range(out_w):
                        w_in = w_out * self.stride

                        receptive_field = X_padded[b, :, h_in:h_in + kernel_h, w_in:w_in + kernel_w]
                        output[b, c_out, h_out, w_out] = np.sum(receptive_field * self.W[c_out]) + self.b[c_out]

        return output

    def backward(self, grads):

        batch_size, out_channels, grad_h, grad_w = grads.shape
        _, in_channels, kernel_h, kernel_w = self.W.shape

        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dX_padded = np.zeros_like(self.input_padded)

        for c_out in range(out_channels):
            db[c_out] = np.sum(grads[:, c_out, :, :])

        for b in range(batch_size):
            for c_out in range(out_channels):
                for h_out in range(grad_h):
                    h_in = h_out * self.stride
                    for w_out in range(grad_w):
                        w_in = w_out * self.stride

                        dW[c_out] += self.input_padded[b, :, h_in:h_in + kernel_h, w_in:w_in + kernel_w] * grads[
                            b, c_out, h_out, w_out]

                        dX_padded[b, :, h_in:h_in + kernel_h, w_in:w_in + kernel_w] += self.W[c_out] * grads[
                            b, c_out, h_out, w_out]

        if self.weight_decay:
            dW += self.weight_decay_lambda * self.W

        self.grads['W'] = dW
        self.grads['b'] = db

        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_padded

        return dX

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}


class ReLU(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X < 0, 0, X)
        return output

    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output


class MultiCrossEntropyLoss(Layer):

    def __init__(self, model=None, max_classes=10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.predicts = None
        self.labels = None
        self.batch_size = None
        self.softmax_output = None
        self.grads = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):

        self.predicts = predicts
        self.labels = labels
        self.batch_size = predicts.shape[0]

        if self.has_softmax:
            self.softmax_output = softmax(predicts)
        else:
            self.softmax_output = predicts

        epsilon = 1e-10
        loss = 0
        for i in range(self.batch_size):
            loss -= np.log(self.softmax_output[i, labels[i]] + epsilon)

        return loss / self.batch_size

    def backward(self):

        self.grads = self.softmax_output.copy()

        for i in range(self.batch_size):
            self.grads[i, self.labels[i]] -= 1

        self.grads /= self.batch_size

        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self


class L2Regularization(Layer):


    def __init__(self, model, lambda_val):
        super().__init__()
        self.model = model
        self.lambda_val = lambda_val

    def forward(self):
        reg_loss = 0
        for layer in self.model.layers:
            if layer.optimizable and hasattr(layer, 'W'):
                reg_loss += 0.5 * self.lambda_val * np.sum(layer.W ** 2)
        return reg_loss

    def backward(self):
        for layer in self.model.layers:
            if layer.optimizable and hasattr(layer, 'W'):
                layer.grads['W'] += self.lambda_val * layer.W


def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition