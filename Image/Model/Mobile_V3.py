import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import torchvision

# 定义可视化函数
def imshow(img):
    img = img / 2 + 0.5  # 反标准化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 主程序入口
if __name__ == '__main__':
    # 设置随机种子，确保结果可复现
    torch.manual_seed(42)

    # 数据路径
    data_dir = 'D:\Data Set\RGB_1'

    # 数据增强和预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),         # 转换为张量
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
    ])

    # 加载完整数据集
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # 数据集划分
    train_size = int(0.7 * len(full_dataset))  # 70% 训练集
    test_size = len(full_dataset) - train_size  # 30% 测试集
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # 数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 打印类别信息
    print("Classes:", full_dataset.classes)  # 类别名称
    print("Class to Index Mapping:", full_dataset.class_to_idx)  # 类别到索引的映射
    print(f"Training set size: {len(train_dataset)}")
    print(f"Testing set size: {len(test_dataset)}")

    # 检查训练集和测试集的类别分布
    train_labels = [full_dataset.targets[idx] for idx in train_dataset.indices]
    test_labels = [full_dataset.targets[idx] for idx in test_dataset.indices]
    print("Training set label distribution:", Counter(train_labels))
    print("Testing set label distribution:", Counter(test_labels))

    # 可视化部分数据
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    print("Labels:", labels)
    # imshow(torchvision.utils.make_grid(images))

    # 初始化 MobileNetV3 模型
    num_classes = len(full_dataset.classes)
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)  # 加载预训练权重
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)  # 修改分类层
    model = model.to('cuda')

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    epochs = 100
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to('cuda'), labels.to('cuda')

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        # 计算训练集准确率
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in train_loader:
                images, labels = images.to('cuda'), labels.to('cuda')
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        train_accuracy = 100 * correct / total

        # 计算测试集准确率
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to('cuda'), labels.to('cuda')
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader)}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")
                
    # 测试模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Test Accuracy: {100 * correct / total:.2f}%")
