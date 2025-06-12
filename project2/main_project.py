# CIFAR-10 Deep Learning Project - Complete Implementation with tqdm
# Neural Network and Deep Learning Course Project 2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import DataLoader
import time
from collections import defaultdict
import seaborn as sns
from matplotlib.lines import Line2D
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_cifar10_loaders(batch_size=128, num_workers=2):
    """Load CIFAR-10 dataset with data augmentation"""

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    return trainloader, testloader


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



class CustomCNN(nn.Module):
    """
    Custom CNN with all required and optional components:
    Required: FC layer, 2D conv, 2D pooling, activations
    Optional: BatchNorm, Dropout, Residual connections
    """

    def __init__(self, num_classes=10, dropout_rate=0.5, use_batchnorm=True, use_residual=True):
        super(CustomCNN, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.use_residual = use_residual

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.pool1 = nn.MaxPool2d(2, 2)  # 2D pooling layer

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128) if use_batchnorm else nn.Identity()

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128) if use_batchnorm else nn.Identity()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256) if use_batchnorm else nn.Identity()

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256) if use_batchnorm else nn.Identity()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)  # Fully-Connected layer
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        if use_residual:
            self.shortcut1 = nn.Conv2d(3, 64, kernel_size=1)  # Input to first block
            self.shortcut2 = nn.Conv2d(64, 128, kernel_size=1)  # First to second block
            self.shortcut3 = nn.Conv2d(128, 256, kernel_size=1)  # Second to third block

    def forward(self, x):
        if self.use_residual:
            identity1 = self.shortcut1(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.use_residual:
            x = F.relu(x + identity1)
        else:
            x = F.relu(x)

        x = self.pool1(x)

        if self.use_residual:
            identity2 = self.shortcut2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))

        if self.use_residual:
            x = F.relu(x + identity2)
        else:
            x = F.relu(x)

        x = self.pool2(x)

        if self.use_residual:
            identity3 = self.shortcut3(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.bn6(self.conv6(x))

        if self.use_residual:
            x = F.relu(x + identity3)
        else:
            x = F.relu(x)

        x = self.pool3(x)
        x = self.adaptive_pool(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class LossFunctions:
    """Different loss functions for experimentation"""

    @staticmethod
    def cross_entropy():
        return nn.CrossEntropyLoss()

    @staticmethod
    def cross_entropy_with_l2(weight_decay=1e-4):
        """Cross entropy with L2 regularization"""
        return nn.CrossEntropyLoss(), weight_decay

    @staticmethod
    def focal_loss(alpha=1, gamma=2):
        """Focal Loss for handling class imbalance"""

        class FocalLoss(nn.Module):
            def __init__(self, alpha=1, gamma=2):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma

            def forward(self, inputs, targets):
                ce_loss = F.cross_entropy(inputs, targets, reduce=False)
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                return focal_loss.mean()

        return FocalLoss(alpha, gamma)

    @staticmethod
    def label_smoothing(smoothing=0.1):
        """Label smoothing cross entropy"""

        class LabelSmoothingLoss(nn.Module):
            def __init__(self, smoothing=0.1):
                super().__init__()
                self.smoothing = smoothing

            def forward(self, inputs, targets):
                log_prob = F.log_softmax(inputs, dim=-1)
                weight = inputs.new_ones(inputs.size()) * self.smoothing / (inputs.size(-1) - 1.)
                weight.scatter_(-1, targets.unsqueeze(-1), (1. - self.smoothing))
                loss = (-weight * log_prob).sum(dim=-1).mean()
                return loss

        return LabelSmoothingLoss(smoothing)



class ActivationFunctions:
    """Different activation functions for experimentation"""

    @staticmethod
    def get_activation(name):
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'mish': lambda x: x * torch.tanh(F.softplus(x))
        }
        return activations.get(name, nn.ReLU())



class CustomOptimizers:
    """Different optimizers for experimentation"""

    @staticmethod
    def get_optimizer(model, optimizer_name, lr=0.001, **kwargs):
        optimizers = {
            'adam': optim.Adam(model.parameters(), lr=lr, **kwargs),
            'sgd': optim.SGD(model.parameters(), lr=lr, momentum=0.9, **kwargs),
            'rmsprop': optim.RMSprop(model.parameters(), lr=lr, **kwargs),
            'adamw': optim.AdamW(model.parameters(), lr=lr, **kwargs),
            'adagrad': optim.Adagrad(model.parameters(), lr=lr, **kwargs),
        }
        return optimizers.get(optimizer_name, optim.Adam(model.parameters(), lr=lr))



def train_model(model, trainloader, testloader, criterion, optimizer, num_epochs=50, device=device):
    """Train the model and return training history with progress bar"""
    model.to(device)

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []


    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = running_loss / len(trainloader)
        train_acc = 100. * correct / total

        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_loss /= len(testloader)
        test_acc = 100. * correct / total

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies
    }


class NetworkInsights:
    """Network interpretation and visualization tools"""

    @staticmethod
    def visualize_filters(model, layer_name='conv1', num_filters=16):
        """Visualize convolutional filters"""
        layer = dict(model.named_modules())[layer_name]

        if isinstance(layer, nn.Conv2d):
            filters = layer.weight.data.cpu()

            fig, axes = plt.subplots(4, 4, figsize=(10, 10))
            for i in range(min(num_filters, 16)):
                ax = axes[i // 4, i % 4]

                if filters.shape[1] == 3:  # RGB
                    filter_img = filters[i].permute(1, 2, 0)
                    filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min())
                    ax.imshow(filter_img)
                else:
                    ax.imshow(filters[i, 0], cmap='gray')

                ax.set_title(f'Filter {i + 1}')
                ax.axis('off')

            plt.suptitle(f'Filters from {layer_name}')
            plt.tight_layout()
            plt.savefig(f'filters_{layer_name}.png', dpi=300, bbox_inches='tight')
            plt.show()

    @staticmethod
    def plot_training_curves(history, title="Training History"):
        """Plot training and validation curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(history['train_losses'], label='Train Loss')
        ax1.plot(history['test_losses'], label='Test Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(history['train_accuracies'], label='Train Accuracy')
        ax2.plot(history['test_accuracies'], label='Test Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy Curves')
        ax2.legend()
        ax2.grid(True)

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(f'training_curves_{title.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def analyze_feature_maps(model, dataloader, device=device):
        """Analyze and visualize feature maps"""
        model.eval()

        inputs, _ = next(iter(dataloader))
        inputs = inputs[:1].to(device)

        feature_maps = {}

        def hook_fn(name):
            def hook(module, input, output):
                feature_maps[name] = output.detach().cpu()

            return hook

        handles = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                handle = module.register_forward_hook(hook_fn(name))
                handles.append(handle)

        with torch.no_grad():
            _ = model(inputs)

        for handle in handles:
            handle.remove()

        for name, fmap in list(feature_maps.items())[:3]:
            fmap = fmap[0]

            fig, axes = plt.subplots(4, 4, figsize=(10, 10))
            for i in range(min(16, fmap.shape[0])):
                ax = axes[i // 4, i % 4]
                ax.imshow(fmap[i], cmap='viridis')
                ax.set_title(f'Channel {i + 1}')
                ax.axis('off')

            plt.suptitle(f'Feature Maps from {name}')
            plt.tight_layout()
            plt.savefig(f'feature_maps_{name.replace(".", "_")}.png', dpi=300, bbox_inches='tight')
            plt.show()



class VGG_A(nn.Module):
    """VGG-A architecture for CIFAR-10 (without BatchNorm)"""

    def __init__(self, num_classes=10):
        super(VGG_A, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG_BatchNorm(nn.Module):
    """VGG-A architecture with Batch Normalization"""

    def __init__(self, num_classes=10):
        super(VGG_BatchNorm, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



class LossLandscapeAnalyzer:
    """Analyze loss landscape with different learning rates"""

    @staticmethod
    def train_with_multiple_lr(model_class, trainloader, testloader, learning_rates, num_epochs=50, save_best=True):
        """Train models with different learning rates and optionally save the best model"""
        results = {}
        best_overall_acc = 0
        best_model_state = None
        best_lr = None

        os.makedirs('models', exist_ok=True)

        for lr in tqdm(learning_rates, desc="Learning Rate Experiments"):
            model = model_class().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

            history = train_model(model, trainloader, testloader, criterion, optimizer, num_epochs)
            results[lr] = history

            best_acc_this_run = max(history['test_accuracies'])
            if save_best and best_acc_this_run > best_overall_acc:
                best_overall_acc = best_acc_this_run
                best_model_state = model.state_dict().copy()
                best_lr = lr

            print(f"✅ LR {lr}: Best Test Acc: {best_acc_this_run:.2f}%")

        if save_best and best_model_state:
            model_name = model_class.__name__
            save_path = f"models/{model_name}_best_lr_{best_lr}.pth"
            torch.save(best_model_state, save_path)
            print(f"💾 Best {model_name} model saved to: {save_path}")
            print(f"📊 Best performance: {best_overall_acc:.2f}% at LR {best_lr}")

        return results

    @staticmethod
    def visualize_loss_landscape(results_without_bn, results_with_bn, learning_rates):
        """Visualize loss landscape comparison"""
        plt.figure(figsize=(12, 8))

        max_losses_no_bn = []
        min_losses_no_bn = []

        for epoch in range(len(list(results_without_bn.values())[0]['train_losses'])):
            epoch_losses = [results_without_bn[lr]['train_losses'][epoch] for lr in learning_rates]
            max_losses_no_bn.append(max(epoch_losses))
            min_losses_no_bn.append(min(epoch_losses))

        max_losses_with_bn = []
        min_losses_with_bn = []

        for epoch in range(len(list(results_with_bn.values())[0]['train_losses'])):
            epoch_losses = [results_with_bn[lr]['train_losses'][epoch] for lr in learning_rates]
            max_losses_with_bn.append(max(epoch_losses))
            min_losses_with_bn.append(min(epoch_losses))

        epochs = range(len(max_losses_no_bn))

        plt.fill_between(epochs, min_losses_no_bn, max_losses_no_bn,
                         alpha=0.3, color='green', label='Standard VGG')
        plt.fill_between(epochs, min_losses_with_bn, max_losses_with_bn,
                         alpha=0.3, color='red', label='Standard VGG + BatchNorm')

        plt.xlabel('Epochs')
        plt.ylabel('Loss Magnitude')
        plt.title('Loss Landscape Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig('loss_landscape_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()



class ModelAnalyzer:
    """Comprehensive model analysis tools"""

    @staticmethod
    def count_parameters(model):
        """Count total parameters in the model"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def measure_inference_time(model, dataloader, device=device, num_batches=10):
        """Measure average inference time"""
        model.eval()
        total_time = 0

        with torch.no_grad():
            for i, (inputs, _) in enumerate(dataloader):
                if i >= num_batches:
                    break

                inputs = inputs.to(device)
                start_time = time.time()
                _ = model(inputs)
                end_time = time.time()
                total_time += (end_time - start_time)

        return total_time / num_batches

    @staticmethod
    def compute_model_stats(model, dataloader):
        """Compute comprehensive model statistics"""
        num_params = ModelAnalyzer.count_parameters(model)
        inference_time = ModelAnalyzer.measure_inference_time(model, dataloader)

        # Model size in MB
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        model_size_mb = (param_size + buffer_size) / 1024 / 1024

        return {
            'parameters': num_params,
            'inference_time': inference_time,
            'model_size_mb': model_size_mb
        }



class BasicBlock(nn.Module):
    """Basic residual block"""

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18_CIFAR(nn.Module):
    """Simplified ResNet-18 for CIFAR-10"""

    def __init__(self, num_classes=10):
        super(ResNet18_CIFAR, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



def run_task1_experiments():

    print("TASK 1: CIFAR-10 Classification Experiments")



    trainloader, testloader = get_cifar10_loaders(batch_size=128)

    experiments = [
        {
            'name': 'Baseline_CNN',
            'model': CustomCNN(use_batchnorm=True, use_residual=True, dropout_rate=0.5),
            'optimizer': 'adam',
            'lr': 0.001,
            'loss': 'cross_entropy'
        },
        {
            'name': 'CNN_with_Focal_Loss',
            'model': CustomCNN(use_batchnorm=True, use_residual=True, dropout_rate=0.5),
            'optimizer': 'adam',
            'lr': 0.001,
            'loss': 'focal'
        },
        {
            'name': 'CNN_with_Label_Smoothing',
            'model': CustomCNN(use_batchnorm=True, use_residual=True, dropout_rate=0.5),
            'optimizer': 'adamw',
            'lr': 0.001,
            'loss': 'label_smoothing'
        }
    ]

    results = {}

    os.makedirs('models', exist_ok=True)

    for exp in tqdm(experiments, desc="Task 1 Experiments"):
        print(f"\nRunning experiment: {exp['name']}")

        model = exp['model']
        optimizer = CustomOptimizers.get_optimizer(model, exp['optimizer'], lr=exp['lr'])

        if exp['loss'] == 'cross_entropy':
            criterion = LossFunctions.cross_entropy()
        elif exp['loss'] == 'focal':
            criterion = LossFunctions.focal_loss()
        elif exp['loss'] == 'label_smoothing':
            criterion = LossFunctions.label_smoothing()

        # Train model
        history = train_model(model, trainloader, testloader, criterion, optimizer, num_epochs=30)
        results[exp['name']] = history


        model_save_path = f"models/{exp['name']}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to: {model_save_path}")

        best_acc = max(history['test_accuracies'])
        final_acc = history['test_accuracies'][-1]
        print(f"{exp['name']}: Best Acc: {best_acc:.2f}%, Final Acc: {final_acc:.2f}%")

        NetworkInsights.plot_training_curves(history, exp['name'])


    model = experiments[0]['model']

    NetworkInsights.visualize_filters(model, 'conv1', 16)

    NetworkInsights.analyze_feature_maps(model, testloader)

    return results


def run_task2_experiments():
    """Run Task 2: Batch Normalization Analysis"""

    print("TASK 2: Batch Normalization Analysis")



    trainloader, testloader = get_cifar10_loaders(batch_size=128)


    learning_rates = [1e-4, 5e-4, 1e-3, 2e-3]

    print("Training VGG-A without Batch Normalization...")
    results_without_bn = LossLandscapeAnalyzer.train_with_multiple_lr(
        VGG_A, trainloader, testloader, learning_rates, num_epochs=30
    )

    print("Training VGG-A with Batch Normalization...")
    results_with_bn = LossLandscapeAnalyzer.train_with_multiple_lr(
        VGG_BatchNorm, trainloader, testloader, learning_rates, num_epochs=30
    )

    print("Generating loss landscape visualization...")
    LossLandscapeAnalyzer.visualize_loss_landscape(results_without_bn, results_with_bn, learning_rates)

    print("Comparing best performing models...")

    best_lr_no_bn = min(learning_rates, key=lambda lr: min(results_without_bn[lr]['test_losses']))
    best_lr_with_bn = min(learning_rates, key=lambda lr: min(results_with_bn[lr]['test_losses']))

    print(f"Best LR without BN: {best_lr_no_bn}")
    print(f"Best LR with BN: {best_lr_with_bn}")

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(results_without_bn[best_lr_no_bn]['train_losses'], label='Without BN', color='green')
    plt.plot(results_with_bn[best_lr_with_bn]['train_losses'], label='With BN', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(results_without_bn[best_lr_no_bn]['test_losses'], label='Without BN', color='green')
    plt.plot(results_with_bn[best_lr_with_bn]['test_losses'], label='With BN', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title('Test Loss Comparison')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(results_without_bn[best_lr_no_bn]['test_accuracies'], label='Without BN', color='green')
    plt.plot(results_with_bn[best_lr_with_bn]['test_accuracies'], label='With BN', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy Comparison')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('vgg_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    return results_without_bn, results_with_bn


def run_comprehensive_experiments():
    """Run all experiments and generate complete analysis"""
    print("=" * 70)
    print("COMPREHENSIVE CIFAR-10 DEEP LEARNING PROJECT")
    print("=" * 70)

    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    trainloader, testloader = get_cifar10_loaders(batch_size=128)


    print("TASK 1: CUSTOM CNN EXPERIMENTS")


    task1_results = run_task1_experiments()


    print("TASK 2: BATCH NORMALIZATION ANALYSIS")

    task2_results = run_task2_experiments()


    print("ADDITIONAL MODEL COMPARISONS")


    models_to_compare = [
        ('Custom_CNN', CustomCNN(use_batchnorm=True, use_residual=True)),
        ('VGG_A', VGG_A()),
        ('VGG_BatchNorm', VGG_BatchNorm()),
        ('ResNet18_CIFAR', ResNet18_CIFAR())
    ]

    comparison_results = {}

    for name, model in tqdm(models_to_compare, desc="Model Comparison"):
        print(f"\nTraining {name}...")
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        history = train_model(model, trainloader, testloader, criterion, optimizer, num_epochs=30)

        stats = ModelAnalyzer.compute_model_stats(model, testloader)

        comparison_results[name] = {
            'history': history,
            'stats': stats,
            'best_test_acc': max(history['test_accuracies'])
        }

        torch.save(model.state_dict(), f'models/{name}.pth')

        print(f"{name} - Best Test Accuracy: {comparison_results[name]['best_test_acc']:.2f}%")
        print(f"Parameters: {stats['parameters']:,}")
        print(f"Model Size: {stats['model_size_mb']:.2f} MB")
        print(f"Inference Time: {stats['inference_time']:.4f} seconds")


    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    for name in comparison_results:
        plt.plot(comparison_results[name]['history']['test_accuracies'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy Comparison')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    for name in comparison_results:
        plt.plot(comparison_results[name]['history']['test_losses'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title('Test Loss Comparison')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    names = list(comparison_results.keys())
    params = [comparison_results[name]['stats']['parameters'] for name in names]
    accuracies = [comparison_results[name]['best_test_acc'] for name in names]

    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    for i, name in enumerate(names):
        plt.scatter(params[i], accuracies[i], s=100, c=[colors[i]], label=name)

    plt.xlabel('Number of Parameters')
    plt.ylabel('Best Test Accuracy (%)')
    plt.title('Parameter Efficiency')
    plt.legend()
    plt.grid(True)


    plt.subplot(2, 2, 4)
    sizes = [comparison_results[name]['stats']['model_size_mb'] for name in names]

    for i, name in enumerate(names):
        plt.scatter(sizes[i], accuracies[i], s=100, c=[colors[i]], label=name)

    plt.xlabel('Model Size (MB)')
    plt.ylabel('Best Test Accuracy (%)')
    plt.title('Model Size vs Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('results/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


    print("\n" + "=" * 50)
    print("EXPERIMENT SUMMARY")
    print("=" * 50)

    print("\nModel Performance Summary:")
    print("-" * 70)
    print(f"{'Model':<20} {'Best Acc (%)':<12} {'Parameters':<12} {'Size (MB)':<10}")
    print("-" * 70)

    for name in comparison_results:
        stats = comparison_results[name]['stats']
        acc = comparison_results[name]['best_test_acc']
        print(f"{name:<20} {acc:<12.2f} {stats['parameters']:<12,} {stats['model_size_mb']:<10.2f}")


    best_model = max(comparison_results.keys(),
                     key=lambda x: comparison_results[x]['best_test_acc'])

    print(f"\nBest performing model: {best_model}")
    print(f"Best test accuracy: {comparison_results[best_model]['best_test_acc']:.2f}%")

    return {
        'task1': task1_results,
        'task2': task2_results,
        'comparison': comparison_results
    }




if __name__ == "__main__":

    torch.manual_seed(42)
    np.random.seed(42)

    results = run_comprehensive_experiments()

    print("\ndone!")