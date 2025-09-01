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
import warnings
warnings.filterwarnings('ignore')

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
        - transform (callable, optional): 一个可选的图像转换函数。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        
        # 检查根目录是否存在
        if not os.path.exists(root_dir):
            raise ValueError(f"数据集根目录不存在: {root_dir}")
        
        # 获取所有类别（子目录）
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        
        if len(self.classes) == 0:
            raise ValueError(f"在 {root_dir} 中没有找到类别子目录")
        
        print(f"找到 {len(self.classes)} 个类别: {self.classes}")
        
        # 支持的图像格式
        valid_extensions = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
        
        # 遍历每个类别
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            file_count = 0
            
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                
                # 检查文件是否为图像文件（修复了原来的 '1.png' 错误）
                if file_name.lower().endswith(valid_extensions) and os.path.isfile(file_path):
                    # 验证文件是否真的存在并且可读
                    try:
                        # 尝试打开图像以验证它是有效的
                        with Image.open(file_path) as img:
                            img.verify()
                        self.data.append(file_path)
                        self.labels.append(label)
                        file_count += 1
                    except Exception as e:
                        print(f"警告: 无法读取图像 {file_path}: {e}")
            
            print(f"  类别 '{class_name}': 找到 {file_count} 张图像")
        
        if len(self.data) == 0:
            raise ValueError("没有找到有效的图像文件！")
        
        print(f"总共加载了 {len(self.data)} 张图像")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            label = self.labels[idx]
            return image, label
        except Exception as e:
            print(f"错误: 无法加载图像 {img_path}: {e}")
            # 返回一个黑色图像作为后备
            if self.transform:
                dummy_image = Image.new('RGB', (224, 224), color='black')
                image = self.transform(dummy_image)
            else:
                image = torch.zeros(3, 224, 224)
            return image, self.labels[idx]


# 图像预处理
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # MobileNet 的输入需求
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet 的均值和标准差
])

# 验证集/测试集的转换（不包含数据增强）
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 定义 MobileNet v2 模型
class MobileNetV2Custom(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.2):
        """
        初始化 MobileNetV2Custom 类。

        参数:
        - num_classes (int): 数据集的类别数量。
        - dropout_rate (float): Dropout 率，用于防止过拟合。
        """
        super(MobileNetV2Custom, self).__init__()
        # 加载预训练的 MobileNet v2 模型
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        
        # 冻结部分层（可选，根据数据集大小调整）
        # for param in self.mobilenet.features[:-4].parameters():
        #     param.requires_grad = False
        
        # 替换分类层
        in_features = self.mobilenet.last_channel
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.mobilenet(x)

# 训练和验证函数
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50):
    train_metrics = {"loss": [], "accuracy": [], "f1_score": []}
    val_metrics = {"loss": [], "accuracy": [], "f1_score": []}
    
    best_val_acc = 0.0
    best_model_wts = None

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        all_train_preds, all_train_labels = [], []

        for batch_idx, (inputs, labels) in enumerate(train_loader):
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
            
            # 打印批次进度
            if batch_idx % 10 == 0:
                print(f"  Batch [{batch_idx}/{len(train_loader)}]", end='\r')

        train_acc = train_correct / train_total
        train_f1 = f1_score(all_train_labels, all_train_preds, average="weighted")
        avg_train_loss = train_loss / len(train_loader)
        
        train_metrics["loss"].append(avg_train_loss)
        train_metrics["accuracy"].append(train_acc)
        train_metrics["f1_score"].append(train_f1)
        
        # 学习率调度
        scheduler.step()

        # 验证阶段
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)
        val_metrics["loss"].append(val_loss)
        val_metrics["accuracy"].append(val_acc)
        val_metrics["f1_score"].append(val_f1)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = model.state_dict().copy()

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"  Train - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print("-" * 60)

    # 加载最佳模型权重
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
        print(f"\n最佳验证准确率: {best_val_acc:.4f}")

    print("\n训练完成！")
    print(f"最终训练指标 - Accuracy: {train_metrics['accuracy'][-1]:.4f}, F1 Score: {train_metrics['f1_score'][-1]:.4f}")
    print(f"最终验证指标 - Accuracy: {val_metrics['accuracy'][-1]:.4f}, F1 Score: {val_metrics['f1_score'][-1]:.4f}")

    return train_metrics, val_metrics

# 评估函数
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
def plot_confusion_matrix(cm, class_names, title, save_path=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=18, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# 绘制训练曲线
def plot_training_curves(train_metrics, val_metrics, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss曲线
    axes[0].plot(train_metrics["loss"], label="Train Loss", linewidth=2)
    axes[0].plot(val_metrics["loss"], label="Validation Loss", linewidth=2)
    axes[0].set_title("Loss Over Epochs", fontsize=16, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=14)
    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy曲线
    axes[1].plot(train_metrics["accuracy"], label="Train Accuracy", linewidth=2)
    axes[1].plot(val_metrics["accuracy"], label="Validation Accuracy", linewidth=2)
    axes[1].set_title("Accuracy Over Epochs", fontsize=16, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # F1 Score曲线
    axes[2].plot(train_metrics["f1_score"], label="Train F1 Score", linewidth=2)
    axes[2].plot(val_metrics["f1_score"], label="Validation F1 Score", linewidth=2)
    axes[2].set_title("F1 Score Over Epochs", fontsize=16, fontweight="bold")
    axes[2].set_xlabel("Epoch", fontsize=14)
    axes[2].set_ylabel("F1 Score", fontsize=14)
    axes[2].legend(fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# 主程序
if __name__ == "__main__":
    # 设置数据路径（请根据实际情况修改）
    data_dir = 'D:/VSCode/fusion/Data/soybean_RGB'
    
    # 验证数据目录
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录不存在: {data_dir}")
        print("请检查路径是否正确！")
        exit(1)
    
    try:
        # 创建数据集（训练集使用数据增强）
        print("加载数据集...")
        full_dataset = CustomImageDataset(root_dir=data_dir, transform=None)
        
        # 数据集划分
        train_ratio = 0.7
        val_ratio = 0.2
        test_ratio = 0.1
        dataset_size = len(full_dataset)
        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)
        test_size = dataset_size - train_size - val_size
        
        print(f"\n数据集划分:")
        print(f"  训练集: {train_size} 张图像")
        print(f"  验证集: {val_size} 张图像")
        print(f"  测试集: {test_size} 张图像")
        
        # 划分数据集
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)  # 设置随机种子以确保可重复性
        )
        
        # 为不同的数据集应用不同的转换
        train_dataset.dataset.transform = image_transform
        val_dataset.dataset = CustomImageDataset(root_dir=data_dir, transform=val_transform)
        test_dataset.dataset = CustomImageDataset(root_dir=data_dir, transform=val_transform)
        
        # 重新划分（使用相同的索引）
        train_indices = train_dataset.indices
        val_indices = val_dataset.indices
        test_indices = test_dataset.indices
        
        # 创建新的数据集实例
        train_dataset = torch.utils.data.Subset(
            CustomImageDataset(root_dir=data_dir, transform=image_transform),
            train_indices
        )
        val_dataset = torch.utils.data.Subset(
            CustomImageDataset(root_dir=data_dir, transform=val_transform),
            val_indices
        )
        test_dataset = torch.utils.data.Subset(
            CustomImageDataset(root_dir=data_dir, transform=val_transform),
            test_indices
        )
        
        # DataLoader
        batch_size = 32
        num_workers = 0  # Windows下设置为0避免多进程问题
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=num_workers)
        
        # 初始化模型、损失函数和优化器
        num_classes = len(full_dataset.classes)
        print(f"\n类别数量: {num_classes}")
        print(f"类别名称: {full_dataset.classes}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        model = MobileNetV2Custom(num_classes=num_classes, dropout_rate=0.2).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        
        # 训练模型
        print("\n开始训练...")
        train_metrics, val_metrics = train_and_validate(
            model, train_loader, val_loader, criterion, optimizer, scheduler, 
            num_epochs=50
        )
        
        # 可视化训练结果
        print("\n绘制训练曲线...")
        plot_training_curves(train_metrics, val_metrics)
        
        # 测试集评估
        print("\n在测试集上评估...")
        test_acc, test_f1, cm = test_model(model, test_loader)
        print(f"测试集准确率: {test_acc:.4f}")
        print(f"测试集 F1 Score: {test_f1:.4f}")
        
        # 绘制混淆矩阵
        plot_confusion_matrix(cm, full_dataset.classes, "Test Set Confusion Matrix")
        
        # 保存模型
        save_model = input("\n是否保存模型? (y/n): ")
        if save_model.lower() == 'y':
            torch.save(model.state_dict(), 'mobilenet_v2_soybean.pth')
            print("模型已保存为 'mobilenet_v2_soybean.pth'")
        
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()