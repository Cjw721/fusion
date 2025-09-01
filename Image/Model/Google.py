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
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                if file_path.endswith(('1.png', '.jpg', '.jpeg')):
                    self.data.append(file_path)
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
    transforms.Resize((224, 224)),  # 调整大小为 GoogLeNet 的输入需求
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 使用 ImageNet 的均值和标准差
])

# 数据加载
image_dataset = CustomImageDataset(root_dir='D:/VSCode/fusion/Data/corn_RGB', transform=image_transform)

# 数据集划分
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2
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

# 定义 GoogLeNet 模型
class GoogLeNetCustom(nn.Module):
    def __init__(self, num_classes):
        super(GoogLeNetCustom, self).__init__()
        self.googlenet = models.googlenet(pretrained=True)
        self.googlenet.fc = nn.Linear(1024, num_classes)  # 替换分类层

    def forward(self, x):
        return self.googlenet(x)

# 初始化模型、损失函数和优化器
num_classes = len(image_dataset.classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GoogLeNetCustom(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

# 训练和验证函数
# 训练和验证函数
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=200):
    train_metrics = {"accuracy": [], "f1_score": [], "loss": []}
    val_metrics = {"accuracy": [], "f1_score": [], "loss": []}

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
        train_metrics["loss"].append(train_loss / len(train_loader))  # 保存训练损失
        scheduler.step()

        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)
        val_metrics["accuracy"].append(val_acc)
        val_metrics["f1_score"].append(val_f1)
        val_metrics["loss"].append(val_loss)  # 保存验证损失

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

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

epochs = range(1, len(val_metrics["accuracy"]) + 1)

fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

# Accuracy
axes[0].plot(epochs, val_metrics["accuracy"], color="royalblue", linewidth=2)
axes[0].set_ylabel("Accuracy", fontsize=16, fontweight="bold", family="Times New Roman")
axes[0].grid(True, linestyle="--", alpha=0.5)

# F1
axes[1].plot(epochs, val_metrics["f1_score"], color="seagreen", linewidth=2)
axes[1].set_ylabel("F1 Score", fontsize=16, fontweight="bold", family="Times New Roman")
axes[1].grid(True, linestyle="--", alpha=0.5)

# Loss
axes[2].plot(epochs, val_metrics["loss"], color="firebrick", linewidth=2)
axes[2].set_ylabel("Loss", fontsize=16, fontweight="bold", family="Times New Roman")
axes[2].set_xlabel("Epoch", fontsize=16, fontweight="bold", family="Times New Roman")
axes[2].grid(True, linestyle="--", alpha=0.5)

# 统一字体大小
for ax in axes:
    ax.tick_params(axis="both", labelsize=14)

plt.suptitle("Validation Metrics Over Epochs", fontsize=18, fontweight="bold", family="Times New Roman")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()



