import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime
import json

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
def visualize_saved_cnn_filters(model_path):
    """
    加载保存的CNN模型并可视化其卷积滤波器

    参数:
    - model_path: 保存的模型路径
    """
    # 创建一个临时配置来初始化模型
    # 这个配置会在加载模型时被替换
    temp_config = {
        "name": "temp",
        "conv1_filters": 4,
        "conv1_kernel_size": 3,
        "conv1_stride": 1,
        "conv1_padding": 1,
        "learning_rate": 0.01,
        "momentum": 0.0,
        "use_weight_decay": False,
        "weight_decay_lambda": 0
    }

    # 创建模型实例
    model = TinyCNN(temp_config)

    # 加载保存的模型参数
    model.load_model(model_path)

    # 获取加载后的配置信息
    config = model.config
    print(f"模型配置: {config['name']}")
    print(f"卷积滤波器数量: {config['conv1_filters']}")
    print(f"卷积核大小: {config['conv1_kernel_size']}x{config['conv1_kernel_size']}")

    # 获取第一个卷积层的权重
    weights = model.conv1.W  # 假设conv1是第一个卷积层的属性名

    # 创建图形显示滤波器
    n_filters = weights.shape[0]
    fig, axes = plt.subplots(1, n_filters, figsize=(n_filters * 2, 2))

    if n_filters == 1:
        axes = [axes]  # 确保axes始终是可迭代的

    for i in range(n_filters):
        # 获取第i个滤波器的第一个通道
        filter_img = weights[i, 0]

        # 归一化显示
        vmin, vmax = filter_img.min(), filter_img.max()
        if vmax > vmin:
            filter_img = (filter_img - vmin) / (vmax - vmin)

        axes[i].imshow(filter_img, cmap='viridis')
        axes[i].set_title(f'Filter {i + 1}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.suptitle(f'CNN  - {config["name"]}', fontsize=14)
    plt.subplots_adjust(top=0.85)

    save_dir = os.path.dirname(model_path)
    save_path = os.path.join(save_dir, f"{config['name']}_filters.png")
    plt.savefig(save_path)
    print(f"滤波器可视化已保存到: {save_path}")

    plt.show()
visualize_saved_cnn_filters(r'C:\Users\apoll\Desktop\university\third\大三下\神经网络\pj2\PJ1-2\codes\tiny_cnn_results\CNN_2_Aug_best.pickle')