import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 设置字体为 Times New Roman 并加粗，字体大小为16
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 16

# 自定义数据集类
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        初始化自定义数据集类。

        参数:
        - root_dir (str): 数据集根目录的路径。
        - transform (callable, optional): 一个可选的图像转换函数，用于在加载图像时对其进行预处理。
        """
        # 存储数据集根目录的路径
        self.root_dir = root_dir
        # 存储图像转换函数
        self.transform = transform
        # 初始化一个空列表来存储图像文件的路径
        self.data = []
        # 初始化一个空列表来存储图像对应的标签
        self.labels = []
        # 获取根目录下的所有子目录，并按字母顺序排序，每个子目录代表一个类别
        self.classes = sorted(os.listdir(root_dir))

        # 遍历每个类别
        for label, class_name in enumerate(self.classes):
            # 构建类别目录的完整路径
            class_dir = os.path.join(root_dir, class_name)
            # 遍历类别目录中的每个文件
            for file_name in os.listdir(class_dir):
                # 构建文件的完整路径
                file_path = os.path.join(class_dir, file_name)
                # 检查文件是否为图像文件（根据文件扩展名）
                if file_path.endswith(('1.png', '.jpg', '.jpeg')):
                    # 将图像文件路径添加到数据列表中
                    self.data.append(file_path)
                    # 将对应的标签添加到标签列表中
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


# 图像预处理
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整大小为 ResNet 的输入需求
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 使用 ImageNet 的均值和标准差
])

# 数据加载
image_dataset = CustomImageDataset(root_dir='D:/VSCode/fusion/Data/RGB_1', transform=image_transform)

# 数据集划分
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1
dataset_size = len(image_dataset)
train_size = int(train_ratio * dataset_size)
val_size = int(val_ratio * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(image_dataset, [train_size, val_size, test_size])

# DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义 ResNet 模型（可以选择不同版本的ResNet）
class ResNetCustom(nn.Module):
    def __init__(self, num_classes, resnet_version='resnet50'):
        """
        初始化 ResNetCustom 类。

        参数:
        - num_classes (int): 数据集的类别数量。
        - resnet_version (str): ResNet版本，可选 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
        """
        super(ResNetCustom, self).__init__()
        
        # 根据版本选择不同的ResNet模型
        if resnet_version == 'resnet18':
            self.resnet = models.resnet18(pretrained=True)
        elif resnet_version == 'resnet34':
            self.resnet = models.resnet34(pretrained=True)
        elif resnet_version == 'resnet50':
            self.resnet = models.resnet50(pretrained=True)
        elif resnet_version == 'resnet101':
            self.resnet = models.resnet101(pretrained=True)
        elif resnet_version == 'resnet152':
            self.resnet = models.resnet152(pretrained=True)
        else:
            raise ValueError(f"Invalid ResNet version: {resnet_version}")
        
        # 获取原始全连接层的输入特征数
        num_features = self.resnet.fc.in_features
        # 替换模型的分类层，以适应新的类别数量
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# 初始化模型、损失函数和优化器
num_classes = len(image_dataset.classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 可以选择不同版本的ResNet：'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
# 这里使用ResNet50作为示例，您可以根据需要更改
model = ResNetCustom(num_classes=num_classes, resnet_version='resnet50').to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

# 训练和验证函数
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50):
    train_metrics = {"accuracy": [], "f1_score": []}
    val_metrics = {"accuracy": [], "f1_score": []}

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        all_train_preds, all_train_labels = [], []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        train_acc = train_correct / train_total
        train_f1 = f1_score(all_train_labels, all_train_preds, average="weighted")
        train_metrics["accuracy"].append(train_acc)
        train_metrics["f1_score"].append(train_f1)
        scheduler.step()

        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)
        val_metrics["accuracy"].append(val_acc)
        val_metrics["f1_score"].append(val_f1)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

    print("\nFinal Training Metrics:")
    print(f"Accuracy: {train_metrics['accuracy'][-1]:.4f}, F1 Score: {train_metrics['f1_score'][-1]:.4f}")
    print("\nFinal Validation Metrics:")
    print(f"Accuracy: {val_metrics['accuracy'][-1]:.4f}, F1 Score: {val_metrics['f1_score'][-1]:.4f}")

    return train_metrics, val_metrics

# 验证函数
def evaluate(model, loader, criterion):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = correct / total
    val_f1 = f1_score(all_labels, all_preds, average="weighted")
    return val_loss / len(loader), val_acc, val_f1

# 测试函数
def test_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)
    return test_acc, test_f1, cm

# 混淆矩阵绘制
def plot_confusion_matrix(cm, class_names, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=14, fontweight="bold")
    plt.show()


# 训练和验证
train_metrics, val_metrics = train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler)

# 可视化训练与验证指标
plt.figure(figsize=(10, 5))
plt.plot(train_metrics["accuracy"], label="Train Accuracy")
plt.plot(val_metrics["accuracy"], label="Validation Accuracy")
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_metrics["f1_score"], label="Train F1 Score")
plt.plot(val_metrics["f1_score"], label="Validation F1 Score")
plt.title("F1 Score Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.legend()
plt.show()

# 测试集评估
test_acc, test_f1, cm = test_model(model, test_loader)
print(f"Test Accuracy: {test_acc:.4f}, Test F1 Score: {test_f1:.4f}")
plot_confusion_matrix(cm, image_dataset.classes, "Test Set Confusion Matrix")