
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime
from scipy.ndimage import rotate, shift
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

    angle = np.random.uniform(-max_angle, max_angle)
    image_2d = image.reshape(28, 28)
    rotated = rotate(image_2d, angle, reshape=False, mode='nearest')
    return rotated.reshape(-1)

def random_shift(image, max_shift=2):
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

def mlp_grid_search():

    (train_imgs, train_labs), (valid_imgs, valid_labs), (test_imgs, test_labs) = load_mnist()

    results_dir = "./mlp_grid_search_results"
    os.makedirs(results_dir, exist_ok=True)

    configurations = [

        {
            "name": "MLP_1_NoAug",
            "hidden_layers": [128, 64],
            "learning_rate": 0.01,
            "momentum": 0.0,
            "weight_decay": [0, 0],
            "batch_size": 32,
            "epochs": 3,
            "use_augmentation": False
        },
        {
            "name": "MLP_2_NoAug",
            "hidden_layers": [256, 128],
            "learning_rate": 0.01,
            "momentum": 0.9,
            "weight_decay": [1e-5, 1e-5],
            "batch_size": 64,
            "epochs": 3,
            "use_augmentation": False
        },
        {
            "name": "MLP_3_NoAug",
            "hidden_layers": [512, 256],
            "learning_rate": 0.005,
            "momentum": 0.9,
            "weight_decay": [1e-4, 1e-4],
            "batch_size": 64,
            "epochs": 3,
            "use_augmentation": False
        },
        {
            "name": "MLP_4_NoAug",
            "hidden_layers": [512, 256, 128],
            "learning_rate": 0.005,
            "momentum": 0.9,
            "weight_decay": [1e-4, 1e-4, 1e-4],
            "batch_size": 128,
            "epochs": 3,
            "use_augmentation": False
        },
        {
            "name": "MLP_5_NoAug",
            "hidden_layers": [1024, 512, 256],
            "learning_rate": 0.001,
            "momentum": 0.95,
            "weight_decay": [1e-4, 1e-4, 1e-4],
            "batch_size": 128,
            "epochs": 3,
            "use_augmentation": False
        },

        {
            "name": "MLP_1_Aug",
            "hidden_layers": [128, 64],
            "learning_rate": 0.01,
            "momentum": 0.0,
            "weight_decay": [0, 0],
            "batch_size": 32,
            "epochs": 3,
            "use_augmentation": True,
            "augment_ratio": 0.5
        },
        {
            "name": "MLP_2_Aug",
            "hidden_layers": [256, 128],
            "learning_rate": 0.01,
            "momentum": 0.9,
            "weight_decay": [1e-5, 1e-5],
            "batch_size": 64,
            "epochs": 3,
            "use_augmentation": True,
            "augment_ratio": 0.5
        },
        {
            "name": "MLP_3_Aug",
            "hidden_layers": [512, 256],
            "learning_rate": 0.005,
            "momentum": 0.9,
            "weight_decay": [1e-4, 1e-4],
            "batch_size": 64,
            "epochs": 3,
            "use_augmentation": True,
            "augment_ratio": 0.7
        },
        {
            "name": "MLP_4_Aug",
            "hidden_layers": [512, 256, 128],
            "learning_rate": 0.005,
            "momentum": 0.9,
            "weight_decay": [1e-4, 1e-4, 1e-4],
            "batch_size": 128,
            "epochs": 3,
            "use_augmentation": True,
            "augment_ratio": 0.7
        },
        {
            "name": "MLP_5_Aug",
            "hidden_layers": [1024, 512, 256],
            "learning_rate": 0.001,
            "momentum": 0.95,
            "weight_decay": [1e-4, 1e-4, 1e-4],
            "batch_size": 128,
            "epochs": 3,
            "use_augmentation": True,
            "augment_ratio": 0.7
        }
    ]

    results = []

    for config in configurations:
        print(f"\n===== Testing configuration: {config['name']} =====")

        size_list = [train_imgs.shape[-1]] + config['hidden_layers'] + [10]
        model = nn.models.Model_MLP(size_list, 'ReLU', config['weight_decay'] + [0])  # No weight decay for output layer

        if config['momentum'] > 0:
            optimizer = nn.optimizer.MomentGD(init_lr=config['learning_rate'], model=model, mu=config['momentum'])
        else:
            optimizer = nn.optimizer.SGD(init_lr=config['learning_rate'], model=model)

        scheduler = nn.lr_scheduler.StepLR(optimizer=optimizer, step_size=1000, gamma=0.5)

        loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)

        batch_size = config['batch_size']
        runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn,
                                  batch_size=batch_size, scheduler=scheduler)


        start_time = datetime.now()

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
                    # Get regular batch
                    start_idx = iteration * batch_size
                    end_idx = min(start_idx + batch_size, X.shape[0])

                    if start_idx >= X.shape[0]:
                        break

                    batch_X = X[start_idx:end_idx]
                    batch_y = y[start_idx:end_idx]

                # Forward pass
                logits = model(batch_X)
                trn_loss = loss_fn(logits, batch_y)
                train_losses.append(trn_loss)

                # Compute accuracy
                trn_score = nn.metric.accuracy(logits, batch_y)
                train_scores.append(trn_score)

                # Backward pass
                loss_fn.backward()

                # Update parameters
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                # Evaluate on validation set (not on every iteration to save time)
                if iteration % 20 == 0:
                    with_batch = min(1000, valid_imgs.shape[0])  # Use a subset for faster evaluation
                    dev_score, dev_loss = runner.evaluate([valid_imgs[:with_batch], valid_labs[:with_batch]])
                    dev_scores.append(dev_score)
                    dev_losses.append(dev_loss)

                    if (iteration) % 100 == 0:
                        elapsed = datetime.now() - start_time
                        print(f"epoch: {epoch}, iteration: {iteration}, time elapsed: {elapsed}")
                        print(f"[Train] loss: {trn_loss:.4f}, score: {trn_score:.4f}")
                        print(f"[Dev] loss: {dev_loss:.4f}, score: {dev_score:.4f}")

                    # Save best model
                    if dev_score > best_score:
                        save_path = os.path.join(results_dir, f"{config['name']}_best.pickle")
                        model.save_model(save_path)
                        print(f"best accuracy updated: {best_score:.5f} --> {dev_score:.5f}")
                        best_score = dev_score

            # Full validation set evaluation at the end of each epoch
            full_dev_score, full_dev_loss = runner.evaluate([valid_imgs, valid_labs])
            print(f"End of epoch {epoch}: Validation accuracy: {full_dev_score:.4f}, loss: {full_dev_loss:.4f}")

            # Save best model
            if full_dev_score > best_score:
                save_path = os.path.join(results_dir, f"{config['name']}_best.pickle")
                model.save_model(save_path)
                print(f"best accuracy updated: {best_score:.5f} --> {full_dev_score:.5f}")
                best_score = full_dev_score

        # Evaluate on test set
        test_score, test_loss = runner.evaluate([test_imgs, test_labs])
        print(f"Test accuracy: {test_score:.4f}, Test loss: {test_loss:.4f}")

        # Save plots
        _, axes = plt.subplots(1, 2, figsize=(12, 5))
        plt.suptitle(f"Training Curves - {config['name']}")
        plot_training_curves(runner, axes)
        plt.savefig(os.path.join(results_dir, f"{config['name']}_plots.png"))
        plt.close()

        # Save config and results
        result = {
            "config": config,
            "best_validation_accuracy": float(best_score),
            "test_accuracy": float(test_score),
            "test_loss": float(test_loss)
        }
        results.append(result)

        # Save individual result
        with open(os.path.join(results_dir, f"{config['name']}_result.json"), 'w') as f:
            json.dump(result, f, indent=4)

    # Save all results
    with open(os.path.join(results_dir, "all_results.json"), 'w') as f:
        json.dump(results, f, indent=4)

    # Print summary
    print("\n===== Grid Search Summary =====")
    print("{:<20} {:<25} {:<15}".format("Configuration", "Validation Accuracy", "Test Accuracy"))
    print("-" * 60)

    for result in sorted(results, key=lambda x: x["test_accuracy"], reverse=True):
        print("{:<20} {:<25.4f} {:<15.4f}".format(
            result["config"]["name"],
            result["best_validation_accuracy"],
            result["test_accuracy"]
        ))

    # Print parameter count summary
    print("\n===== Model Parameter Counts =====")
    print("{:<20} {:<15}".format("Configuration", "Parameter Count"))
    print("-" * 40)

    for config in configurations:
        # Calculate parameter count
        param_count = 0

        # Input layer parameters
        in_dim = 784  # 28*28

        # Hidden layers
        for i, out_dim in enumerate(config["hidden_layers"]):
            if i == 0:
                # First hidden layer (from input)
                param_count += in_dim * out_dim + out_dim  # weights + biases
            else:
                # Subsequent hidden layers
                param_count += config["hidden_layers"][i-1] * out_dim + out_dim  # weights + biases

        # Output layer
        param_count += config["hidden_layers"][-1] * 10 + 10  # weights + biases for 10 classes

        print("{:<20} {:<15,d}".format(config["name"], param_count))

if __name__ == "__main__":
    mlp_grid_search()