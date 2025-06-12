# 评估已保存模型的准确率
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomCNN(nn.Module):
    """自定义CNN模型"""

    def __init__(self, num_classes=10, dropout_rate=0.5, use_batchnorm=True, use_residual=True):
        super(CustomCNN, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.use_residual = use_residual

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.pool1 = nn.MaxPool2d(2, 2)


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
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)


        if use_residual:
            self.shortcut1 = nn.Conv2d(3, 64, kernel_size=1)
            self.shortcut2 = nn.Conv2d(64, 128, kernel_size=1)
            self.shortcut3 = nn.Conv2d(128, 256, kernel_size=1)

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


class VGG_A(nn.Module):
    """VGG-A架构（无BatchNorm）"""

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
    """VGG-A架构（有BatchNorm）"""

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


class BasicBlock(nn.Module):
    """ResNet基础块"""

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
    """ResNet-18 for CIFAR-10"""

    def __init__(self, num_classes=10):
        super(ResNet18_CIFAR, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
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



def get_test_loader():
    """获取测试数据集"""
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    return testloader



def evaluate_model(model, testloader, device):
    """评估模型准确率"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def evaluate_all_saved_models():
    """评估所有保存的模型"""
    print("=" * 60)
    print("评估所有已保存的模型")
    print("=" * 60)


    testloader = get_test_loader()
    model_configs = [
        ("Baseline_CNN.pth", CustomCNN, {"use_batchnorm": True, "use_residual": True, "dropout_rate": 0.5}),
        ("CNN_with_Focal_Loss.pth", CustomCNN, {"use_batchnorm": True, "use_residual": True, "dropout_rate": 0.5}),
        ("CNN_with_Label_Smoothing.pth", CustomCNN, {"use_batchnorm": True, "use_residual": True, "dropout_rate": 0.5}),
        ("Custom_CNN.pth", CustomCNN, {"use_batchnorm": True, "use_residual": True}),
        ("VGG_A.pth", VGG_A, {}),
        ("VGG_A_best_lr_0.002.pth", VGG_A, {}),
        ("VGG_BatchNorm.pth", VGG_BatchNorm, {}),
        ("VGG_BatchNorm_best_lr_0.002.pth", VGG_BatchNorm, {}),
        ("ResNet18_CIFAR_Fixed.pth", ResNet18_CIFAR, {}),
    ]

    results = []

    for model_file, model_class, model_params in model_configs:
        model_path = os.path.join("models", model_file)

        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_file}")
            continue

        try:
            print(f"\n评估模型: {model_file}")


            model = model_class(**model_params)

            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)

            accuracy = evaluate_model(model, testloader, device)

            params = count_parameters(model)

            result = {
                'model_name': model_file.replace('.pth', ''),
                'accuracy': accuracy,
                'parameters': params
            }
            results.append(result)

            print(f"{model_file}: {accuracy:.2f}% accuracy, {params:,} parameters")

        except Exception as e:
            print(f"评估 {model_file} 时出错: {e}")

    print("\n" + "=" * 60)
    print("模型性能汇总")
    print("=" * 60)
    print(f"{'模型名称':<25} {'准确率':<10} {'参数数量':<12}")
    print("-" * 60)

    results.sort(key=lambda x: x['accuracy'], reverse=True)

    for result in results:
        print(f"{result['model_name']:<25} {result['accuracy']:<10.2f}% {result['parameters']:<12,}")

    if results:
        best_model = results[0]
        print(f"\n最佳模型: {best_model['model_name']}")
        print(f"   准确率: {best_model['accuracy']:.2f}%")
        print(f"   参数量: {best_model['parameters']:,}")

    return results


if __name__ == "__main__":
    print(f"Using device: {device}")
    results = evaluate_all_saved_models()