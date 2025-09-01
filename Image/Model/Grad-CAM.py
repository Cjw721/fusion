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
import numpy as np
import cv2
from torch.nn import functional as F
from datetime import datetime

# 设置字体为 Times New Roman 并加粗，字体大小为16
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 16

# 创建保存结果的文件夹
def create_save_directories():
    """创建用于保存结果的文件夹结构"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"results_{timestamp}"
    
    dirs = {
        'base': base_dir,
        'training': os.path.join(base_dir, 'training_curves'),
        'confusion': os.path.join(base_dir, 'confusion_matrix'),
        'gradcam': os.path.join(base_dir, 'gradcam'),
        'gradcam_single': os.path.join(base_dir, 'gradcam', 'single_images'),
        'gradcam_batch': os.path.join(base_dir, 'gradcam', 'batch_visualization'),
        'gradcam_class': os.path.join(base_dir, 'gradcam', 'class_analysis'),
        'model': os.path.join(base_dir, 'saved_model')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"Created result directories in: {base_dir}")
    return dirs

# 自定义数据集类
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        初始化自定义数据集类。

        参数:
        - root_dir (str): 数据集根目录的路径。
        - transform (callable, optional): 一个可选的图像转换函数，用于在加载图像时对其进行预处理。
        """
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
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Grad-CAM的预处理（不包含数据增强）
gradcam_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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

# 定义 MobileNet v2 模型
class MobileNetV2Custom(nn.Module):
    def __init__(self, num_classes):
        """
        初始化 MobileNetV2Custom 类。

        参数:
        - num_classes (int): 数据集的类别数量。
        """
        super(MobileNetV2Custom, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.mobilenet.classifier[1] = nn.Linear(self.mobilenet.last_channel, num_classes)

    def forward(self, x):
        return self.mobilenet(x)


# Grad-CAM 实现
class GradCAM:
    def __init__(self, model, target_layer):
        """
        初始化Grad-CAM
        
        参数:
        - model: 训练好的模型
        - target_layer: 目标层，用于生成CAM
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        self.handles = []
        self.handles.append(target_layer.register_forward_hook(self.save_activation))
        self.handles.append(target_layer.register_backward_hook(self.save_gradient))
        
    def save_activation(self, module, input, output):
        """保存前向传播的特征图"""
        self.activations = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output):
        """保存反向传播的梯度"""
        self.gradients = grad_output[0].detach()
        
    def generate_cam(self, input_image, class_idx=None):
        """
        生成CAM热力图
        
        参数:
        - input_image: 输入图像张量
        - class_idx: 目标类别索引，如果为None则使用预测类别
        """
        self.model.eval()
        
        # 前向传播
        output = self.model(input_image)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # 反向传播
        self.model.zero_grad()
        class_score = output[:, class_idx].squeeze()
        class_score.backward(retain_graph=True)
        
        # 计算权重
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # 生成CAM
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # 归一化
        cam = cam.squeeze()
        if len(cam.shape) > 2:
            cam = cam[0]
        
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
        
        return cam.cpu().numpy()
    
    def remove_hooks(self):
        """移除钩子"""
        for handle in self.handles:
            handle.remove()


def visualize_gradcam(model, image_path, target_layer, class_names, device, save_dir=None, image_name=None):
    """
    可视化单个图像的Grad-CAM结果
    
    参数:
    - model: 训练好的模型
    - image_path: 图像路径
    - target_layer: 目标层
    - class_names: 类别名称列表
    - device: 设备
    - save_dir: 保存目录
    - image_name: 保存的图像名称
    """
    # 加载原始图像
    original_image = Image.open(image_path).convert('RGB')
    original_image_np = np.array(original_image.resize((224, 224)))
    
    # 预处理图像
    input_tensor = gradcam_transform(original_image).unsqueeze(0).to(device)
    
    # 创建Grad-CAM实例
    gradcam = GradCAM(model, target_layer)
    
    # 生成CAM
    cam = gradcam.generate_cam(input_tensor)
    
    # 获取预测结果
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        pred_prob = torch.softmax(output, dim=1)[0, pred_class].item()
    
    # 将CAM调整到原始图像大小
    cam_resized = cv2.resize(cam, (224, 224))
    
    # 生成热力图
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # 叠加图像
    superimposed = heatmap * 0.4 + original_image_np * 0.6
    superimposed = np.uint8(superimposed)
    
    # 可视化 - 增大图像尺寸和字体
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(original_image_np)
    axes[0].set_title('Original Image', fontsize=16, fontweight='bold', pad=15)
    axes[0].axis('off')
    
    axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=16, fontweight='bold', pad=15)
    axes[1].axis('off')
    
    axes[2].imshow(superimposed)
    axes[2].set_title(f'Prediction: {class_names[pred_class]} ({pred_prob:.2%})', 
                     fontsize=16, fontweight='bold', pad=15)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    if save_dir:
        if image_name is None:
            image_name = f"gradcam_{os.path.basename(image_path).split('.')[0]}"
        save_path = os.path.join(save_dir, f"{image_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
        print(f"Saved: {save_path}")
        
        # 同时保存各个组件
        Image.fromarray(original_image_np).save(os.path.join(save_dir, f"{image_name}_original.png"))
        plt.imsave(os.path.join(save_dir, f"{image_name}_heatmap.png"), cam_resized, cmap='jet')
        Image.fromarray(superimposed).save(os.path.join(save_dir, f"{image_name}_overlay.png"))
    
    plt.show()
    
    # 清理钩子
    gradcam.remove_hooks()
    
    return cam, pred_class, pred_prob


def visualize_gradcam_batch(model, test_loader, target_layer, class_names, device, save_dir=None, num_samples=6):
    """
    批量可视化Grad-CAM结果
    
    参数:
    - model: 训练好的模型
    - test_loader: 测试数据加载器
    - target_layer: 目标层
    - class_names: 类别名称列表
    - device: 设备
    - save_dir: 保存目录
    - num_samples: 要可视化的样本数量
    """
    model.eval()
    gradcam = GradCAM(model, target_layer)
    
    # 获取一批测试数据
    images, labels = next(iter(test_loader))
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]
    
    # 创建子图 - 增大图像尺寸
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # 用于保存单独的图像
    individual_results = []
    
    for idx in range(num_samples):
        # 获取单个图像
        input_tensor = images[idx:idx+1]
        true_label = labels[idx].item()
        
        # 生成CAM
        cam = gradcam.generate_cam(input_tensor)
        
        # 获取预测结果
        with torch.no_grad():
            output = model(input_tensor)
            pred_class = output.argmax(dim=1).item()
            pred_prob = torch.softmax(output, dim=1)[0, pred_class].item()
        
        # 反归一化图像用于显示
        img = images[idx].cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        img_np = img.permute(1, 2, 0).numpy()
        
        # 调整CAM大小
        cam_resized = cv2.resize(cam, (224, 224))
        
        # 生成热力图
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        # 叠加图像
        superimposed = heatmap * 0.4 + img_np * 0.6
        
        # 保存单独的结果
        if save_dir:
            individual_results.append({
                'original': (img_np * 255).astype(np.uint8),
                'heatmap': cam_resized,
                'overlay': (superimposed * 255).astype(np.uint8),
                'true_label': class_names[true_label],
                'pred_label': class_names[pred_class],
                'pred_prob': pred_prob,
                'correct': pred_class == true_label
            })
        
        # 显示 - 增大字体
        axes[idx, 0].imshow(img_np)
        axes[idx, 0].set_title(f'Original\nTrue: {class_names[true_label]}', fontsize=14, fontweight='bold')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(cam_resized, cmap='jet')
        axes[idx, 1].set_title('Grad-CAM', fontsize=14, fontweight='bold')
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(superimposed)
        axes[idx, 2].set_title(f'Overlay\nPred: {class_names[pred_class]}', fontsize=14, fontweight='bold')
        axes[idx, 2].axis('off')
        
        # 显示预测概率条形图
        with torch.no_grad():
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        top_k = min(5, len(class_names))
        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_probs = probs[top_indices]
        top_labels = [class_names[i] for i in top_indices]
        
        axes[idx, 3].barh(range(top_k), top_probs)
        axes[idx, 3].set_yticks(range(top_k))
        axes[idx, 3].set_yticklabels(top_labels, fontsize=12)
        axes[idx, 3].set_xlabel('Probability', fontsize=12, fontweight='bold')
        axes[idx, 3].set_title(f'Top {top_k} Predictions', fontsize=14, fontweight='bold')
        axes[idx, 3].set_xlim([0, 1])
        
        # 标记正确预测
        if pred_class == true_label:
            color = 'green'
            marker = '✓'
        else:
            color = 'red'
            marker = '✗'
        axes[idx, 2].text(0.05, 0.95, marker, transform=axes[idx, 2].transAxes,
                          fontsize=24, color=color, fontweight='bold')
    
    plt.suptitle('Grad-CAM Visualization Results', fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    # 保存批量可视化结果
    if save_dir:
        batch_save_path = os.path.join(save_dir, f'batch_visualization_{num_samples}_samples.png')
        plt.savefig(batch_save_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
        print(f"Saved batch visualization: {batch_save_path}")
        
        # 保存每个样本的单独图像
        for i, result in enumerate(individual_results):
            prefix = f"sample_{i+1}_{'correct' if result['correct'] else 'wrong'}"
            
            Image.fromarray(result['original']).save(
                os.path.join(save_dir, f"{prefix}_original.png"))
            plt.imsave(os.path.join(save_dir, f"{prefix}_heatmap.png"), 
                      result['heatmap'], cmap='jet')
            Image.fromarray(result['overlay']).save(
                os.path.join(save_dir, f"{prefix}_overlay.png"))
        
        print(f"Saved {len(individual_results)} individual sample visualizations")
    
    plt.show()
    
    # 清理钩子
    gradcam.remove_hooks()


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
def plot_confusion_matrix(cm, class_names, title, save_path=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names,
                annot_kws={"fontsize": 14, "fontweight": "bold"})
    plt.title(title, fontsize=18, fontweight="bold", pad=20)
    plt.xlabel("Predicted Label", fontsize=16, fontweight="bold")
    plt.ylabel("True Label", fontsize=16, fontweight="bold")
    plt.xticks(fontsize=14, fontweight="bold")
    plt.yticks(fontsize=14, fontweight="bold")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
        print(f"Saved confusion matrix: {save_path}")
    
    plt.show()


# 主程序
if __name__ == "__main__":
    # 创建保存目录
    save_dirs = create_save_directories()
    
    # 初始化模型、损失函数和优化器
    num_classes = len(image_dataset.classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = MobileNetV2Custom(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # 训练和验证
    print("="*50)
    print("Starting Training and Validation")
    print("="*50)
    train_metrics, val_metrics = train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler)

    # 可视化训练与验证指标 - Accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(train_metrics["accuracy"], label="Train Accuracy", linewidth=2.5)
    plt.plot(val_metrics["accuracy"], label="Validation Accuracy", linewidth=2.5)
    plt.title("Accuracy Over Epochs", fontsize=18, fontweight="bold", pad=15)
    plt.xlabel("Epoch", fontsize=16, fontweight="bold")
    plt.ylabel("Accuracy", fontsize=16, fontweight="bold")
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    accuracy_plot_path = os.path.join(save_dirs['training'], 'accuracy_curve.png')
    plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    print(f"Saved accuracy plot: {accuracy_plot_path}")
    plt.show()

    # 可视化训练与验证指标 - F1 Score
    plt.figure(figsize=(12, 6))
    plt.plot(train_metrics["f1_score"], label="Train F1 Score", linewidth=2.5)
    plt.plot(val_metrics["f1_score"], label="Validation F1 Score", linewidth=2.5)
    plt.title("F1 Score Over Epochs", fontsize=18, fontweight="bold", pad=15)
    plt.xlabel("Epoch", fontsize=16, fontweight="bold")
    plt.ylabel("F1 Score", fontsize=16, fontweight="bold")
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    f1_plot_path = os.path.join(save_dirs['training'], 'f1_score_curve.png')
    plt.savefig(f1_plot_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    print(f"Saved F1 score plot: {f1_plot_path}")
    plt.show()

    # 测试集评估
    print("\n" + "="*50)
    print("Testing Model Performance")
    print("="*50)
    test_acc, test_f1, cm = test_model(model, test_loader)
    print(f"Test Accuracy: {test_acc:.4f}, Test F1 Score: {test_f1:.4f}")
    
    # 保存混淆矩阵
    cm_path = os.path.join(save_dirs['confusion'], 'confusion_matrix.png')
    plot_confusion_matrix(cm, image_dataset.classes, "Test Set Confusion Matrix", save_path=cm_path)
    
    # 保存测试结果到文本文件
    results_file = os.path.join(save_dirs['base'], 'test_results.txt')
    with open(results_file, 'w') as f:
        f.write("Model Test Results\n")
        f.write("="*50 + "\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test F1 Score: {test_f1:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nClass Names:\n")
        for i, class_name in enumerate(image_dataset.classes):
            f.write(f"{i}: {class_name}\n")
    print(f"Saved test results: {results_file}")
    
    # ==================== Grad-CAM 可视化部分 ====================
    print("\n" + "="*50)
    print("Starting Grad-CAM Visualization")
    print("="*50)
    
    # 获取MobileNetV2的最后一个卷积层
    target_layer = model.mobilenet.features[-1][0]
    
    # 方法1: 可视化多个单独的测试样本（前5个）
    test_indices = test_dataset.indices
    print("\nVisualizing individual test samples...")
    for i in range(min(5, len(test_indices))):
        idx = test_indices[i]
        image_path = image_dataset.data[idx]
        true_label = image_dataset.labels[idx]
        
        print(f"Processing sample {i+1}: {os.path.basename(image_path)}")
        visualize_gradcam(
            model=model,
            image_path=image_path,
            target_layer=target_layer,
            class_names=image_dataset.classes,
            device=device,
            save_dir=save_dirs['gradcam_single'],
            image_name=f"sample_{i+1}_true_{image_dataset.classes[true_label]}"
        )
    
    # 方法2: 批量可视化多个测试样本
    print("\nVisualizing batch of test samples...")
    visualize_gradcam_batch(
        model=model,
        test_loader=test_loader,
        target_layer=target_layer,
        class_names=image_dataset.classes,
        device=device,
        save_dir=save_dirs['gradcam_batch'],
        num_samples=6
    )
    
    # 方法3: 分析不同类别的激活模式 - 改进版本
    print("\nAnalyzing activation patterns for different classes...")
    
    # 增大图像尺寸和改进布局
    fig, axes = plt.subplots(len(image_dataset.classes), 3, 
                            figsize=(15, 5*len(image_dataset.classes)))
    
    if len(image_dataset.classes) == 1:
        axes = axes.reshape(1, -1)
    
    model.eval()
    gradcam = GradCAM(model, target_layer)
    
    for class_idx, class_name in enumerate(image_dataset.classes):
        found = False
        for i in test_indices[:50]:
            if image_dataset.labels[i] == class_idx:
                sample_path = image_dataset.data[i]
                
                # 加载和预处理图像
                original_image = Image.open(sample_path).convert('RGB')
                original_image_np = np.array(original_image.resize((224, 224)))
                input_tensor = gradcam_transform(original_image).unsqueeze(0).to(device)
                
                # 生成CAM
                cam = gradcam.generate_cam(input_tensor, class_idx=class_idx)
                
                # 获取预测
                with torch.no_grad():
                    output = model(input_tensor)
                    pred_class = output.argmax(dim=1).item()
                    pred_prob = torch.softmax(output, dim=1)[0, pred_class].item()
                
                # 调整CAM大小并生成热力图
                cam_resized = cv2.resize(cam, (224, 224))
                heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                superimposed = heatmap * 0.4 + original_image_np * 0.6
                superimposed = np.uint8(superimposed)
                
                # 保存单独的类别分析结果
                class_save_dir = os.path.join(save_dirs['gradcam_class'], f"class_{class_name}")
                os.makedirs(class_save_dir, exist_ok=True)
                
                Image.fromarray(original_image_np).save(
                    os.path.join(class_save_dir, f"{class_name}_original.png"))
                plt.imsave(os.path.join(class_save_dir, f"{class_name}_heatmap.png"), 
                          cam_resized, cmap='jet')
                Image.fromarray(superimposed).save(
                    os.path.join(class_save_dir, f"{class_name}_overlay.png"))
                
                # 显示结果 - 改进的文字显示
                axes[class_idx, 0].imshow(original_image_np)
                axes[class_idx, 0].set_title(f'{class_name} - Original', 
                                           fontsize=14, fontweight='bold', pad=10)
                axes[class_idx, 0].axis('off')
                
                axes[class_idx, 1].imshow(cam_resized, cmap='jet')
                axes[class_idx, 1].set_title('Grad-CAM', 
                                           fontsize=14, fontweight='bold', pad=10)
                axes[class_idx, 1].axis('off')
                
                axes[class_idx, 2].imshow(superimposed)
                axes[class_idx, 2].set_title(f'Pred: {image_dataset.classes[pred_class]} ({pred_prob:.2%})', 
                                           fontsize=14, fontweight='bold', pad=10)
                axes[class_idx, 2].axis('off')
                
                found = True
                break
        
        if not found:
            for j in range(3):
                axes[class_idx, j].axis('off')
                axes[class_idx, j].text(0.5, 0.5, f'No sample found for {class_name}',
                                       ha='center', va='center', 
                                       transform=axes[class_idx, j].transAxes,
                                       fontsize=12)
    
    # 改进的标题和布局设置
    plt.suptitle('Grad-CAM Analysis by Class', fontsize=18, fontweight='bold', y=0.995)
    
    # 调整子图间距
    plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.03, 
                       hspace=0.15, wspace=0.1)
    
    # 保存类别分析总图
    class_analysis_path = os.path.join(save_dirs['gradcam_class'], 'all_classes_analysis.png')
    plt.savefig(class_analysis_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    print(f"Saved class analysis: {class_analysis_path}")
    plt.show()
    
    gradcam.remove_hooks()
    
    # 额外创建单独的类别可视化（每个类别一张图）
    print("\nCreating individual class visualizations...")
    
    for class_idx, class_name in enumerate(image_dataset.classes):
        fig_individual = plt.figure(figsize=(18, 6))
        
        found = False
        for i in test_indices[:50]:
            if image_dataset.labels[i] == class_idx:
                sample_path = image_dataset.data[i]
                
                # 加载和预处理图像
                original_image = Image.open(sample_path).convert('RGB')
                original_image_np = np.array(original_image.resize((224, 224)))
                input_tensor = gradcam_transform(original_image).unsqueeze(0).to(device)
                
                # 创建新的GradCAM实例
                gradcam_individual = GradCAM(model, target_layer)
                
                # 生成CAM
                cam = gradcam_individual.generate_cam(input_tensor, class_idx=class_idx)
                
                # 获取预测
                with torch.no_grad():
                    output = model(input_tensor)
                    pred_class = output.argmax(dim=1).item()
                    pred_prob = torch.softmax(output, dim=1)[0, pred_class].item()
                
                # 调整CAM大小并生成热力图
                cam_resized = cv2.resize(cam, (224, 224))
                heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
                superimposed = heatmap * 0.4 + original_image_np / 255.0
                
                # 创建子图
                ax1 = plt.subplot(1, 3, 1)
                ax1.imshow(original_image_np)
                ax1.set_title(f'{class_name} - Original', fontsize=16, fontweight='bold', pad=15)
                ax1.axis('off')
                
                ax2 = plt.subplot(1, 3, 2)
                ax2.imshow(cam_resized, cmap='jet')
                ax2.set_title('Grad-CAM Heatmap', fontsize=16, fontweight='bold', pad=15)
                ax2.axis('off')
                
                ax3 = plt.subplot(1, 3, 3)
                ax3.imshow(superimposed)
                prediction_text = f'Prediction: {image_dataset.classes[pred_class]}\nConfidence: {pred_prob:.1%}'
                ax3.set_title(prediction_text, fontsize=16, fontweight='bold', pad=15)
                ax3.axis('off')
                
                # 添加类别标签作为主标题
                plt.suptitle(f'Class: {class_name}', fontsize=20, fontweight='bold', y=1.02)
                
                plt.tight_layout()
                
                # 保存单独的类别图像
                individual_save_path = os.path.join(save_dirs['gradcam_class'], 
                                                   f'class_{class_name}_analysis_individual.png')
                plt.savefig(individual_save_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
                print(f"Saved individual analysis for {class_name}: {individual_save_path}")
                
                plt.show()
                
                # 清理
                gradcam_individual.remove_hooks()
                
                found = True
                break
        
        if not found:
            print(f"No sample found for class: {class_name}")
        
        plt.close(fig_individual)
    
    # 保存模型
    model_path = os.path.join(save_dirs['model'], 'mobilenet_v2_trained.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'classes': image_dataset.classes
    }, model_path)
    print(f"\nModel and metrics saved: {model_path}")
    
    # 创建总结报告
    summary_file = os.path.join(save_dirs['base'], 'summary_report.txt')
    with open(summary_file, 'w') as f:
        f.write("MobileNetV2 with Grad-CAM - Training and Analysis Report\n")
        f.write("="*60 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Number of Classes: {num_classes}\n")
        f.write(f"Classes: {', '.join(image_dataset.classes)}\n\n")
        
        f.write("Dataset Split:\n")
        f.write(f"- Training samples: {train_size}\n")
        f.write(f"- Validation samples: {val_size}\n")
        f.write(f"- Test samples: {test_size}\n\n")
        
        f.write("Training Configuration:\n")
        f.write(f"- Batch size: {batch_size}\n")
        f.write(f"- Learning rate: 0.001\n")
        f.write(f"- Weight decay: 1e-4\n")
        f.write(f"- Epochs: 50\n\n")
        
        f.write("Final Results:\n")
        f.write(f"- Training Accuracy: {train_metrics['accuracy'][-1]:.4f}\n")
        f.write(f"- Training F1 Score: {train_metrics['f1_score'][-1]:.4f}\n")
        f.write(f"- Validation Accuracy: {val_metrics['accuracy'][-1]:.4f}\n")
        f.write(f"- Validation F1 Score: {val_metrics['f1_score'][-1]:.4f}\n")
        f.write(f"- Test Accuracy: {test_acc:.4f}\n")
        f.write(f"- Test F1 Score: {test_f1:.4f}\n\n")
        
        f.write("Files Generated:\n")
        f.write(f"- Results saved in: {save_dirs['base']}\n")
        f.write(f"- Training curves: {save_dirs['training']}\n")
        f.write(f"- Confusion matrix: {save_dirs['confusion']}\n")
        f.write(f"- Grad-CAM visualizations: {save_dirs['gradcam']}\n")
        f.write(f"- Trained model: {save_dirs['model']}\n")
    
    print(f"\nSummary report saved: {summary_file}")
    
    print("\n" + "="*50)
    print("All visualizations and results have been saved!")
    print(f"Check the folder: {save_dirs['base']}")
    print("="*50)