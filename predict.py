
import numpy as np
import matplotlib.pyplot as plt
import pickle
from struct import unpack
import gzip
import os
from mynn.models import Model_MLP
from test_CNN import TinyCNN


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


def softmax(X):

    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition


def visualize_prediction(image, true_label, pred_class, probs):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    image_2d = image.reshape(28, 28)
    ax1.imshow(image_2d, cmap='gray')
    ax1.set_title(f'真实标签: {true_label}, 预测: {pred_class}')
    ax1.axis('off')

    bars = ax2.bar(range(10), probs)
    ax2.set_xticks(range(10))
    ax2.set_xlabel('数字类别')
    ax2.set_ylabel('预测概率')
    ax2.set_title('预测概率分布')

    if pred_class == true_label:
        bars[pred_class].set_color('green')
    else:
        bars[true_label].set_color('green')
        bars[pred_class].set_color('red')

    plt.tight_layout()
    plt.show()


def evaluate_model(model, test_images, test_labels, is_cnn=False, num_samples=None):

    if num_samples is None:
        num_samples = len(test_images)
    else:
        num_samples = min(num_samples, len(test_images))

    images = test_images[:num_samples]
    labels = test_labels[:num_samples]

    if is_cnn and images.shape[1] == 784:
        images = images.reshape(num_samples, 1, 28, 28)

    logits = model(images)

    predicted = np.argmax(logits, axis=1)

    accuracy = np.mean(predicted == labels)
    conf_matrix = np.zeros((10, 10), dtype=int)
    for i in range(num_samples):
        conf_matrix[labels[i], predicted[i]] += 1

    return accuracy, predicted, conf_matrix

def main():
    test_images, test_labels = load_mnist_test()
    mlp_model_path = './mlp_grid_search_results/MLP_2_Aug_best.pickle'
    mlp_model = Model_MLP()
    mlp_model.load_model(mlp_model_path)
    print("MLP模型加载完成，开始评估...")

    mlp_accuracy, mlp_predictions, mlp_conf_matrix = evaluate_model(
        mlp_model, test_images, test_labels, is_cnn=False
    )

    print(f"MLP模型准确率: {mlp_accuracy:.4f}")
    cnn_model_path = './tiny_cnn_results/CNN_2_Aug_best.pickle'

    temp_config = {
        "name": "temp",
        "conv1_filters": 16,
        "conv1_kernel_size": 5,
        "conv1_stride": 1,
        "conv1_padding": 2,
        "learning_rate": 0.005,
        "momentum": 0.9,
        "use_weight_decay": True,
        "weight_decay_lambda": 1e-4
    }
    cnn_model = TinyCNN(temp_config)
    cnn_model.load_model(cnn_model_path)

    print("CNN模型加载完成，开始评估...")
    cnn_accuracy, cnn_predictions, cnn_conf_matrix = evaluate_model(
        cnn_model, test_images, test_labels, is_cnn=True
    )

    print(f"CNN模型准确率: {cnn_accuracy:.4f}")
    indices = np.random.choice(len(test_images), 5, replace=False)

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