
import numpy as np
import matplotlib.pyplot as plt
import pickle
from struct import unpack
import gzip
import os
import time

from test_CNN import TinyCNN
from mynn.models import Model_MLP

def softmax(X):

    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition

def load_mnist_test():

    test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
    test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

    with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

    with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

    test_imgs = test_imgs / 255.0

    return test_imgs, test_labs


def predict_single_image(model, image, is_cnn=False):


    if image.ndim == 1:

        image = image.reshape(1, -1)


    if is_cnn and image.shape[1] == 784:
        image = image.reshape(image.shape[0], 1, 28, 28)

    logits = model(image)


    probs = softmax(logits)

    pred_class = np.argmax(probs, axis=1)[0]

    return pred_class, probs[0]

def visualize_prediction(image, true_label, pred_class, probs):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    image_2d = image.reshape(28, 28)
    ax1.imshow(image_2d, cmap='gray')
    ax1.set_title(f'real: {true_label}, predict: {pred_class}')
    ax1.axis('off')

    bars = ax2.bar(range(10), probs)
    ax2.set_xticks(range(10))
    ax2.set_xlabel('number type')
    ax2.set_ylabel('prediction')
    ax2.set_title('prediction distribution')

    if pred_class == true_label:
        bars[pred_class].set_color('green')
    else:
        bars[true_label].set_color('green')
        bars[pred_class].set_color('red')

    plt.tight_layout()
    plt.show()



def evaluate_model(model, test_images, test_labels, is_cnn=False, num_samples=None):

    start_time = time.time()

    if num_samples is None:
        num_samples = len(test_images)
    else:
        num_samples = min(num_samples, len(test_images))

    images = test_images[:num_samples]
    labels = test_labels[:num_samples]

    print(f"准备数据完成，样本数: {num_samples}, 耗时: {time.time() - start_time:.4f}秒")
    reshape_time = time.time()

    if is_cnn and images.shape[1] == 784:
        images = images.reshape(num_samples, 1, 28, 28)
        print(f"输入图像形状: {images.shape}")

    print(f"重塑数据完成，耗时: {time.time() - reshape_time:.4f}秒")

    batch_size = 100
    num_batches = int(np.ceil(num_samples / batch_size))

    all_predictions = []
    correct_count = 0
    forward_time = time.time()

    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, num_samples)
        batch_images = images[batch_start:batch_end]
        batch_labels = labels[batch_start:batch_end]

        batch_start_time = time.time()
        print(f"处理批次 {i + 1}/{num_batches}, 大小: {len(batch_images)}")

        logits = model(batch_images)
        batch_predictions = np.argmax(logits, axis=1)
        all_predictions.extend(batch_predictions)
        correct_count += np.sum(batch_predictions == batch_labels)

        print(f"批次 {i + 1} 完成，耗时: {time.time() - batch_start_time:.4f}秒")
    all_predictions = np.array(all_predictions)
    accuracy = correct_count / num_samples
    conf_matrix = np.zeros((10, 10), dtype=int)
    for i in range(num_samples):
        conf_matrix[labels[i], all_predictions[i]] += 1

    print(f"前向传播完成，耗时: {time.time() - forward_time:.4f}秒")
    print(f"总评估耗时: {time.time() - start_time:.4f}秒")

    return accuracy, all_predictions, conf_matrix


def visualize_cnn_filters(model):
    weights = model.conv1.W

    n_filters = weights.shape[0]
    fig, axes = plt.subplots(2, n_filters // 2, figsize=(12, 5))
    axes = axes.flatten()

    for i in range(n_filters):
        filter_img = weights[i, 0]
        vmin, vmax = filter_img.min(), filter_img.max()
        if vmax > vmin:
            filter_img = (filter_img - vmin) / (vmax - vmin)

        axes[i].imshow(filter_img, cmap='viridis')
        axes[i].set_title(f'Filter {i + 1}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('cnn_filters.png')
    plt.show()


def main():

    test_images, test_labels = load_mnist_test()
    print(f"加载了 {len(test_images)} 张测试图像")

    print("\n===== MLP模型预测 =====")
    mlp_model_path = './mlp_grid_search_results/MLP_2_Aug_best.pickle'
    mlp_model = Model_MLP()
    mlp_model.load_model(mlp_model_path)

    mlp_accuracy, mlp_predictions, mlp_conf_matrix = evaluate_model(
        mlp_model, test_images[:1000], test_labels[:1000], is_cnn=False
    )

    print(f"MLP模型准确率: {mlp_accuracy:.4f}")
    print("\n===== CNN模型预测 =====")
    cnn_model_path = './tiny_cnn_results/CNN_2_Aug_best.pickle'
    temp_config = {
        "name": "CNN_2_Aug",
        "conv1_filters": 8,
        "conv1_kernel_size": 3,
        "conv1_stride": 1,
        "conv1_padding": 1,
        "learning_rate": 0.005,
        "momentum": 0.9,
        "use_weight_decay": True,
        "weight_decay_lambda": 1e-5,
        "batch_size": 64,
        "epochs": 2,
        "use_augmentation": True,
        "augment_ratio": 0.5
    }

    cnn_model = TinyCNN(temp_config)
    cnn_model.load_model(cnn_model_path)

    visualize_cnn_filters(cnn_model)

    print("开始CNN模型评估...")
    cnn_accuracy, cnn_predictions, cnn_conf_matrix = evaluate_model(
        cnn_model, test_images, test_labels, is_cnn=True
    )

    print(f"CNN模型准确率: {cnn_accuracy:.4f}")

    print("\n===== 可视化预测示例 =====")
    indices = np.random.choice(100, 2, replace=False)  # 从前100张图像中选择

    for idx in indices:
        image = test_images[idx]
        true_label = test_labels[idx]
        mlp_pred, mlp_probs = predict_single_image(mlp_model, image, is_cnn=False)
        print(f"样本 {idx}: MLP预测 = {mlp_pred}, 真实标签 = {true_label}")
        visualize_prediction(image, true_label, mlp_pred, mlp_probs)
        cnn_pred, cnn_probs = predict_single_image(cnn_model, image, is_cnn=True)
        print(f"样本 {idx}: CNN预测 = {cnn_pred}, 真实标签 = {true_label}")
        visualize_prediction(image, true_label, cnn_pred, cnn_probs)


if __name__ == "__main__":
    main()