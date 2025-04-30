from .op import *
import pickle


class Model_MLP(Layer):

    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i + 2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)

    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W': layer.params['W'], 'b': layer.params['b'], 'weight_decay': layer.weight_decay,
                                   'lambda': layer.weight_decay_lambda})

        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)


class Model_CNN(Layer):

    def __init__(self, in_channels=1, weight_decay=False, weight_decay_lambda=1e-4):
        super().__init__()
        self.layers = []

        self.layers.append(conv2D(in_channels=in_channels, out_channels=16, kernel_size=5,
                                  stride=1, padding=2, weight_decay=weight_decay,
                                  weight_decay_lambda=weight_decay_lambda))
        self.layers.append(ReLU())

        self.layers.append(conv2D(in_channels=16, out_channels=32, kernel_size=5,
                                  stride=1, padding=2, weight_decay=weight_decay,
                                  weight_decay_lambda=weight_decay_lambda))
        self.layers.append(ReLU())

        self.fc_input_dim = 32 * 28 * 28

        self.layers.append(Linear(in_dim=self.fc_input_dim, out_dim=128,
                                  weight_decay=weight_decay,
                                  weight_decay_lambda=weight_decay_lambda))
        self.layers.append(ReLU())

        self.layers.append(Linear(in_dim=128, out_dim=10,
                                  weight_decay=weight_decay,
                                  weight_decay_lambda=weight_decay_lambda))

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):

        batch_size = X.shape[0]
        if len(X.shape) == 2:

            X = X.reshape(batch_size, 1, 28, 28)

        outputs = X

        for i, layer in enumerate(self.layers):

            if isinstance(layer, Linear) and len(outputs.shape) > 2:
                outputs = outputs.reshape(batch_size, -1)
            outputs = layer(outputs)

        return outputs

    def backward(self, loss_grad):
        batch_size = loss_grad.shape[0]
        grads = loss_grad


        for i, layer in enumerate(reversed(self.layers)):

            if i > 0 and isinstance(self.layers[-(i)], Linear) and isinstance(self.layers[-(i + 1)], conv2D):
                grads = grads.reshape(batch_size, 32, 28, 28)  # Reshape to match conv output
            grads = layer.backward(grads)

        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            params = pickle.load(f)


        for i, layer_params in enumerate(params):
            if i < len(self.layers) and hasattr(self.layers[i], 'params'):
                for key in layer_params:
                    if key in self.layers[i].params:
                        self.layers[i].params[key] = layer_params[key]

                        setattr(self.layers[i], key, layer_params[key])

    def save_model(self, save_path):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'params'):
                layer_params = {}
                for key in layer.params:
                    layer_params[key] = layer.params[key]
                if hasattr(layer, 'weight_decay'):
                    layer_params['weight_decay'] = layer.weight_decay
                    layer_params['lambda'] = layer.weight_decay_lambda
                params.append(layer_params)

        with open(save_path, 'wb') as f:
            pickle.dump(params, f)