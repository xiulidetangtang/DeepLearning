
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime
import json

np.random.seed(309)

def load_mnist():

    train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
    train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'
    test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
    test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

    with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)

    with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)

    with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)

    with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

    idx = np.random.permutation(np.arange(train_imgs.shape[0]))
    train_imgs = train_imgs[idx]
    train_labs = train_labs[idx]
    valid_imgs = train_imgs[:10000]
    valid_labs = train_labs[:10000]
    train_imgs = train_imgs[10000:]
    train_labs = train_labs[10000:]


    train_imgs = train_imgs / 255.0
    valid_imgs = valid_imgs / 255.0
    test_imgs = test_imgs / 255.0

    return (train_imgs, train_labs), (valid_imgs, valid_labs), (test_imgs, test_labs)


def random_rotation(image, max_angle=15):

    from scipy.ndimage import rotate
    angle = np.random.uniform(-max_angle, max_angle)
    image_2d = image.reshape(28, 28)
    rotated = rotate(image_2d, angle, reshape=False, mode='nearest')
    return rotated.reshape(-1)

def random_shift(image, max_shift=2):

    from scipy.ndimage import shift
    dx, dy = np.random.randint(-max_shift, max_shift + 1, size=2)
    image_2d = image.reshape(28, 28)
    shifted = shift(image_2d, [dy, dx], mode='constant', cval=0)
    return shifted.reshape(-1)

def add_noise(image, noise_factor=0.1):

    noise = np.random.randn(*image.shape) * noise_factor
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)

def generate_batch_with_augmentation(X, y, batch_size, augment_ratio=0.5):

    batch_size = min(batch_size, len(X))

    indices = np.random.choice(len(X), batch_size, replace=False)
    X_batch = X[indices].copy()
    y_batch = y[indices].copy()

    num_to_augment = int(batch_size * augment_ratio)

    for i in range(num_to_augment):
        aug_type = np.random.randint(3)

        if aug_type == 0:
            X_batch[i] = random_rotation(X_batch[i])
        elif aug_type == 1:
            X_batch[i] = random_shift(X_batch[i])
        else:
            X_batch[i] = add_noise(X_batch[i])

    return X_batch, y_batch


def plot_training_curves(runner, axes):

    train_scores = runner.train_scores
    dev_scores = runner.dev_scores
    train_losses = runner.train_loss
    dev_losses = runner.dev_loss

    axes[0].plot(train_scores, label='Train')
    if len(dev_scores) > 0:

        axes[0].plot([0] * (len(train_scores) - len(dev_scores)) + dev_scores, label='Validation')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[1].plot(train_losses, label='Train')
    if len(dev_losses) > 0:

        axes[1].plot([0] * (len(train_losses) - len(dev_losses)) + dev_losses, label='Validation')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Loss')
    axes[1].legend()


class TinyCNN(nn.op.Layer):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.layers = []

        self.conv1 = nn.op.conv2D(
            in_channels=1,
            out_channels=config["conv1_filters"],
            kernel_size=config["conv1_kernel_size"],
            stride=config["conv1_stride"],
            padding=config["conv1_padding"],
            weight_decay=config["use_weight_decay"],
            weight_decay_lambda=config["weight_decay_lambda"]
        )
        self.layers.append(self.conv1)

        if config["conv1_stride"] == 1 and config["conv1_padding"] == 1 and config["conv1_kernel_size"] == 3:

            h_out = 28
            w_out = 28
        elif config["conv1_stride"] == 2 and config["conv1_padding"] == 1 and config["conv1_kernel_size"] == 3:

            h_out = 14
            w_out = 14
        elif config["conv1_stride"] == 1 and config["conv1_padding"] == 2 and config["conv1_kernel_size"] == 5:

            h_out = 28
            w_out = 28
        else:

            h_out = (28 + 2*config["conv1_padding"] - config["conv1_kernel_size"]) // config["conv1_stride"] + 1
            w_out = h_out


        self.h_out = h_out
        self.w_out = w_out


        self.fc_input_size = config["conv1_filters"] * h_out * w_out


        self.fc = nn.op.Linear(
            in_dim=self.fc_input_size,
            out_dim=10,
            weight_decay=config["use_weight_decay"],
            weight_decay_lambda=config["weight_decay_lambda"]
        )
        self.layers.append(self.fc)

        self.conv_output = None
        self.flatten_output = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):

        batch_size = X.shape[0]

        if len(X.shape) == 2:
            # 从 [batch, 784] 转为 [batch, 1, 28, 28]
            X = X.reshape(batch_size, 1, 28, 28)


        self.conv_output = self.conv1(X)
        self.conv_output = np.maximum(0, self.conv_output)

        self.flatten_output = self.conv_output.reshape(batch_size, -1)

        output = self.fc(self.flatten_output)

        return output

    def backward(self, grad):

        batch_size = grad.shape[0]
        fc_grad = self.fc.backward(grad)

        conv_grad = fc_grad.reshape(batch_size, self.config["conv1_filters"], self.h_out, self.w_out)
        relu_mask = self.conv_output > 0
        conv_grad = conv_grad * relu_mask

        input_grad = self.conv1.backward(conv_grad)

        return input_grad

    def save_model(self, save_path):

        params = []
        conv_params = {
            'W': self.conv1.params['W'],
            'b': self.conv1.params['b'],
            'weight_decay': self.conv1.weight_decay,
            'lambda': self.conv1.weight_decay_lambda
        }
        params.append(conv_params)

        fc_params = {
            'W': self.fc.params['W'],
            'b': self.fc.params['b'],
            'weight_decay': self.fc.weight_decay,
            'lambda': self.fc.weight_decay_lambda
        }
        params.append(fc_params)

        params.append(self.config)

        with open(save_path, 'wb') as f:
            pickle.dump(params, f)

    def load_model(self, param_path):

        with open(param_path, 'rb') as f:
            params = pickle.load(f)
        conv_params = params[0]
        self.conv1.params['W'] = conv_params['W']
        self.conv1.params['b'] = conv_params['b']
        self.conv1.W = conv_params['W']
        self.conv1.b = conv_params['b']
        self.conv1.weight_decay = conv_params['weight_decay']
        self.conv1.weight_decay_lambda = conv_params['lambda']

        fc_params = params[1]
        self.fc.params['W'] = fc_params['W']
        self.fc.params['b'] = fc_params['b']
        self.fc.W = fc_params['W']
        self.fc.b = fc_params['b']
        self.fc.weight_decay = fc_params['weight_decay']
        self.fc.weight_decay_lambda = fc_params['lambda']

        self.config = params[2]

def visualize_filters(model, save_path):

    weights = model.conv1.W

    n_filters = min(9, weights.shape[0])
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    axes = axes.flatten()

    for i in range(n_filters):

        f = weights[i, 0]

        f_min, f_max = f.min(), f.max()
        if f_max > f_min:
            f = (f - f_min) / (f_max - f_min)

        axes[i].imshow(f, cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f'Filter {i+1}')

    for i in range(n_filters, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def tiny_cnn_grid_search():

    (train_imgs, train_labs), (valid_imgs, valid_labs), (test_imgs, test_labs) = load_mnist()

    results_dir = "./tiny_cnn_results"
    os.makedirs(results_dir, exist_ok=True)

    configurations = [

        {
            "name": "CNN_Aug",
            "conv1_filters": 8,
            "conv1_kernel_size": 3,
            "conv1_stride": 1,
            "conv1_padding": 1,
            "learning_rate": 0.005,
            "momentum": 0.9,
            "use_weight_decay": True,
            "weight_decay_lambda": 1e-5,
            "batch_size": 8,
            "epochs": 1,
            "use_augmentation": True,
            "augment_ratio": 0.5
        }
    ]

    results = []

    for config in configurations:
        print(f"\n===== 测试配置: {config['name']} =====")

        model = TinyCNN(config)

        if config['momentum'] > 0:
            optimizer = nn.optimizer.MomentGD(init_lr=config['learning_rate'], model=model, mu=config['momentum'])
        else:
            optimizer = nn.optimizer.SGD(init_lr=config['learning_rate'], model=model)

        scheduler = nn.lr_scheduler.StepLR(optimizer=optimizer, step_size=500, gamma=0.5)

        loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)

        batch_size = config['batch_size']
        runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn,
                                  batch_size=batch_size, scheduler=scheduler)

        train_scores = []
        dev_scores = []
        train_losses = []
        dev_losses = []

        runner.train_scores = train_scores
        runner.dev_scores = dev_scores
        runner.train_loss = train_losses
        runner.dev_loss = dev_losses

        best_score = 0

        epochs = config['epochs']
        start_time = datetime.now()

        for epoch in range(epochs):
            idx = np.random.permutation(range(train_imgs.shape[0]))
            X = train_imgs[idx]
            y = train_labs[idx]

            num_batches = int(np.ceil(X.shape[0] / batch_size))

            for iteration in range(num_batches):

                if config['use_augmentation']:
                    batch_X, batch_y = generate_batch_with_augmentation(
                        X, y,
                        min(batch_size, X.shape[0] - iteration * batch_size),
                        config['augment_ratio']
                    )
                else:

                    start_idx = iteration * batch_size
                    end_idx = min(start_idx + batch_size, X.shape[0])

                    if start_idx >= X.shape[0]:
                        break

                    batch_X = X[start_idx:end_idx]
                    batch_y = y[start_idx:end_idx]

                logits = model(batch_X)
                trn_loss = loss_fn(logits, batch_y)
                train_losses.append(trn_loss)

                trn_score = nn.metric.accuracy(logits, batch_y)
                train_scores.append(trn_score)

                loss_fn.backward()

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                if iteration % 50 == 0:
                    with_batch = min(1000, valid_imgs.shape[0])
                    dev_score, dev_loss = runner.evaluate([valid_imgs[:with_batch], valid_labs[:with_batch]])
                    dev_scores.append(dev_score)
                    dev_losses.append(dev_loss)

                    if (iteration) % 100 == 0:
                        elapsed = datetime.now() - start_time
                        print(f"epoch: {epoch}, iteration: {iteration}, 已用时间: {elapsed}")
                        print(f"[Train] loss: {trn_loss:.4f}, score: {trn_score:.4f}")
                        print(f"[Dev] loss: {dev_loss:.4f}, score: {dev_score:.4f}")

                    if dev_score > best_score:
                        save_path = os.path.join(results_dir, f"{config['name']}_best.pickle")
                        model.save_model(save_path)
                        print(f"最佳准确率已更新: {best_score:.5f} --> {dev_score:.5f}")
                        best_score = dev_score

            full_dev_score, full_dev_loss = runner.evaluate([valid_imgs, valid_labs])
            print(f"Epoch {epoch} 结束: 验证准确率: {full_dev_score:.4f}, 损失: {full_dev_loss:.4f}")

            if full_dev_score > best_score:
                save_path = os.path.join(results_dir, f"{config['name']}_best.pickle")
                model.save_model(save_path)
                print(f"最佳准确率已更新: {best_score:.5f} --> {full_dev_score:.5f}")
                best_score = full_dev_score

        test_score, test_loss = runner.evaluate([test_imgs, test_labs])
        print(f"测试准确率: {test_score:.4f}, 测试损失: {test_loss:.4f}")

        if not os.path.exists(os.path.join(results_dir, "filters")):
            os.makedirs(os.path.join(results_dir, "filters"))

        visualize_filters(model, os.path.join(results_dir, "filters", f"{config['name']}_filters.png"))

        _, axes = plt.subplots(1, 2, figsize=(12, 5))
        plt.suptitle(f"训练曲线 - {config['name']}")
        plot_training_curves(runner, axes)
        plt.savefig(os.path.join(results_dir, f"{config['name']}_plots.png"))
        plt.close()

        result = {
            "config": config,
            "best_validation_accuracy": float(best_score),
            "test_accuracy": float(test_score),
            "test_loss": float(test_loss)
        }
        results.append(result)
        with open(os.path.join(results_dir, f"{config['name']}_result.json"), 'w') as f:
            json.dump(result, f, indent=4)

    with open(os.path.join(results_dir, "all_results.json"), 'w') as f:
        json.dump(results, f, indent=4)

    print("\n===== 网格搜索摘要 =====")
    print("{:<20} {:<25} {:<15}".format("配置", "验证准确率", "测试准确率"))
    print("-" * 60)

    for result in sorted(results, key=lambda x: x["test_accuracy"], reverse=True):
        print("{:<20} {:<25.4f} {:<15.4f}".format(
            result["config"]["name"],
            result["best_validation_accuracy"],
            result["test_accuracy"]
        ))

    print("\n===== 模型参数数量 =====")
    print("{:<20} {:<15}".format("配置", "参数数量"))
    print("-" * 40)

    for config in configurations:

        param_count = 0

        conv_params = config["conv1_filters"] * (1 * config["conv1_kernel_size"]**2 + 1)
        param_count += conv_params

        if config["conv1_stride"] == 1 and config["conv1_padding"] == 1 and config["conv1_kernel_size"] == 3:

            h_out = 28
            w_out = 28
        elif config["conv1_stride"] == 2 and config["conv1_padding"] == 1 and config["conv1_kernel_size"] == 3:

            h_out = 14
            w_out = 14
        elif config["conv1_stride"] == 1 and config["conv1_padding"] == 2 and config["conv1_kernel_size"] == 5:
            h_out = 28
            w_out = 28
        else:

            h_out = (28 + 2*config["conv1_padding"] - config["conv1_kernel_size"]) // config["conv1_stride"] + 1
            w_out = h_out

        fc_input_size = config["conv1_filters"] * h_out * w_out
        fc_params = 10 * (fc_input_size + 1)
        param_count += fc_params

        print("{:<20} {:<15,d}".format(config["name"], param_count))

if __name__ == "__main__":
    tiny_cnn_grid_search()