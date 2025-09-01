"""
光谱数据深度学习分析 - MobileNet版本
使用轻量级MobileNet网络架构处理光谱数据
包含SG平滑、SNV标准化等预处理方法
"""
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.signal import savgol_filter

# 设置字体为 Times New Roman 并加粗，字体大小为16
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 16


# 光谱预处理函数
class SpectralPreprocessor:
    """光谱数据预处理类，包含SG平滑和SNV标准化"""

    @staticmethod
    def sg_smooth(spectra, window_length=11, polyorder=2, deriv=0):
        """
        Savitzky-Golay平滑滤波

        参数:
        - spectra: 光谱数据 (n_samples, n_features)
        - window_length: 窗口长度（必须为奇数）
        - polyorder: 多项式阶数
        - deriv: 导数阶数（0表示平滑，1表示一阶导数，2表示二阶导数）

        返回:
        - 平滑后的光谱数据
        """
        if window_length % 2 == 0:
            window_length += 1  # 确保窗口长度为奇数

        # 确保窗口长度不超过特征数
        n_features = spectra.shape[1]
        if window_length > n_features:
            window_length = n_features if n_features % 2 == 1 else n_features - 1

        smoothed_spectra = np.zeros_like(spectra)
        for i in range(spectra.shape[0]):
            smoothed_spectra[i] = savgol_filter(
                spectra[i],
                window_length=window_length,
                polyorder=polyorder,
                deriv=deriv
            )
        return smoothed_spectra

    @staticmethod
    def snv_transform(spectra):
        """
        Standard Normal Variate (SNV) 标准正态变量变换
        用于消除光谱数据中的散射效应

        参数:
        - spectra: 光谱数据 (n_samples, n_features)

        返回:
        - SNV变换后的光谱数据
        """
        # 对每个样本进行SNV变换
        snv_spectra = np.zeros_like(spectra)
        for i in range(spectra.shape[0]):
            # 计算每个光谱的均值和标准差
            mean_spectrum = np.mean(spectra[i])
            std_spectrum = np.std(spectra[i])

            # 避免除零错误
            if std_spectrum == 0:
                snv_spectra[i] = spectra[i] - mean_spectrum
            else:
                snv_spectra[i] = (spectra[i] - mean_spectrum) / std_spectrum

        return snv_spectra

    @staticmethod
    def msc_transform(spectra):
        """
        Multiplicative Scatter Correction (MSC) 多元散射校正
        可选的预处理方法

        参数:
        - spectra: 光谱数据 (n_samples, n_features)

        返回:
        - MSC校正后的光谱数据
        """
        # 计算平均光谱
        mean_spectrum = np.mean(spectra, axis=0)

        # MSC校正
        msc_spectra = np.zeros_like(spectra)
        for i in range(spectra.shape[0]):
            # 线性回归拟合
            fit = np.polyfit(mean_spectrum, spectra[i], 1)
            # 校正
            msc_spectra[i] = (spectra[i] - fit[1]) / fit[0]

        return msc_spectra


# 自定义表格数据集类
class TabularDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# 定义MobileNet风格的一维网络模型（适配光谱数据）
class DepthwiseSeparableConv1d(nn.Module):
    """深度可分离卷积块 - MobileNet的核心组件"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv1d, self).__init__()
        # 深度卷积（Depthwise Convolution）
        self.depthwise = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels  # 关键：groups=in_channels实现深度卷积
        )
        self.depthwise_bn = nn.BatchNorm1d(in_channels)

        # 逐点卷积（Pointwise Convolution）
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.pointwise_bn = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 深度卷积
        x = self.depthwise(x)
        x = self.depthwise_bn(x)
        x = self.relu(x)

        # 逐点卷积
        x = self.pointwise(x)
        x = self.pointwise_bn(x)
        x = self.relu(x)

        return x


class MobileNet_1D(nn.Module):
    """MobileNet V1架构的一维版本，适用于光谱数据"""

    def __init__(self, input_dim, num_classes, width_mult=1.0, dropout_rate=0.2):
        """
        参数:
        - input_dim: 输入特征维度
        - num_classes: 分类数
        - width_mult: 宽度乘数，用于调整网络宽度（默认1.0）
        - dropout_rate: Dropout率
        """
        super(MobileNet_1D, self).__init__()

        # 定义基础通道数
        base_channels = 32
        channels = int(base_channels * width_mult)

        # 第一层：标准卷积
        self.conv1 = nn.Conv1d(1, channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

        # MobileNet层配置 (输入通道, 输出通道, stride)
        self.mobilenet_config = [
            # Block 1
            (channels, int(64 * width_mult), 1),
            # Block 2
            (int(64 * width_mult), int(128 * width_mult), 2),
            (int(128 * width_mult), int(128 * width_mult), 1),
            # Block 3
            (int(128 * width_mult), int(256 * width_mult), 2),
            (int(256 * width_mult), int(256 * width_mult), 1),
            # Block 4
            (int(256 * width_mult), int(512 * width_mult), 2),
            # Block 5-10 (6个重复的512通道块)
            (int(512 * width_mult), int(512 * width_mult), 1),
            (int(512 * width_mult), int(512 * width_mult), 1),
            (int(512 * width_mult), int(512 * width_mult), 1),
            (int(512 * width_mult), int(512 * width_mult), 1),
            (int(512 * width_mult), int(512 * width_mult), 1),
            # Block 11
            (int(512 * width_mult), int(1024 * width_mult), 2),
            # Block 12
            (int(1024 * width_mult), int(1024 * width_mult), 1),
        ]

        # 构建深度可分离卷积层
        self.layers = nn.ModuleList()
        for in_c, out_c, s in self.mobilenet_config:
            self.layers.append(
                DepthwiseSeparableConv1d(
                    in_c, out_c,
                    kernel_size=3,
                    stride=s,
                    padding=1
                )
            )

        # 自适应平均池化
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # 分类器
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(int(1024 * width_mult), num_classes)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # 将输入reshape为(batch_size, 1, input_dim)格式
        x = x.unsqueeze(1)  # 添加通道维度

        # 第一层标准卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # 深度可分离卷积层
        for layer in self.layers:
            x = layer(x)

        # 全局平均池化
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        # 分类器
        x = self.dropout(x)
        x = self.fc(x)

        return x


class MobileNetV2Block(nn.Module):
    """MobileNet V2的反向残差块（可选的增强版本）"""

    def __init__(self, in_channels, out_channels, expansion=6, stride=1):
        super(MobileNetV2Block, self).__init__()

        hidden_dim = in_channels * expansion
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []

        # Expansion layer
        if expansion != 1:
            layers.extend([
                nn.Conv1d(in_channels, hidden_dim, kernel_size=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])

        # Depthwise convolution
        layers.extend([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3,
                      stride=stride, padding=1, groups=hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])

        # Projection layer
        layers.extend([
            nn.Conv1d(hidden_dim, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


# 数据预处理和加载
def load_and_preprocess_data(file_path,
                             apply_sg=True,
                             sg_window=11,
                             sg_polyorder=2,
                             sg_deriv=0,
                             apply_snv=True,
                             apply_msc=False):
    """
    从Excel文件加载数据并进行预处理
    假设第一列是标签，其余列是特征

    参数:
    - file_path: Excel文件路径
    - apply_sg: 是否应用SG平滑
    - sg_window: SG平滑窗口长度
    - sg_polyorder: SG多项式阶数
    - sg_deriv: SG导数阶数
    - apply_snv: 是否应用SNV变换
    - apply_msc: 是否应用MSC校正（与SNV二选一）
    """
    # 读取Excel文件
    df = pd.read_excel(file_path)

    # 分离特征和标签
    labels = df.iloc[:, 0].values  # 第一列为标签
    features = df.iloc[:, 1:].values  # 其余列为特征

    print("=" * 60)
    print("原始数据信息:")
    print(f"数据形状: {features.shape}")
    print(f"特征数量: {features.shape[1]}")
    print(f"样本数量: {features.shape[0]}")

    # 创建预处理器实例
    preprocessor = SpectralPreprocessor()

    # 应用光谱预处理
    processed_features = features.copy()

    # 1. SG平滑
    if apply_sg:
        print(f"\n应用SG平滑 (窗口={sg_window}, 多项式阶数={sg_polyorder}, 导数阶数={sg_deriv})...")
        processed_features = preprocessor.sg_smooth(
            processed_features,
            window_length=sg_window,
            polyorder=sg_polyorder,
            deriv=sg_deriv
        )
        print("SG平滑完成")

    # 2. SNV或MSC变换（二选一）
    if apply_snv:
        print("\n应用SNV标准化...")
        processed_features = preprocessor.snv_transform(processed_features)
        print("SNV标准化完成")
    elif apply_msc:
        print("\n应用MSC校正...")
        processed_features = preprocessor.msc_transform(processed_features)
        print("MSC校正完成")

    # 3. 额外的特征标准化（可选）
    # 注意：SNV已经进行了标准化，这一步可能不必要
    # 但为了与原代码保持一致，保留这个选项
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(processed_features)

    # 处理标签编码
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    print("\n预处理后的数据信息:")
    print(f"特征形状: {scaled_features.shape}")
    print(f"类别数量: {len(label_encoder.classes_)}")
    print(f"类别标签: {label_encoder.classes_}")
    print("=" * 60)

    return scaled_features, encoded_labels, label_encoder, scaler, processed_features


# 绘制预处理效果对比图
def plot_preprocessing_comparison(original, processed, sample_indices=[0, 1, 2]):
    """
    绘制预处理前后的光谱对比图

    参数:
    - original: 原始光谱数据
    - processed: 预处理后的光谱数据
    - sample_indices: 要显示的样本索引列表
    """
    n_samples = len(sample_indices)
    fig, axes = plt.subplots(n_samples, 2, figsize=(15, 5 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    wavelengths = np.arange(original.shape[1])

    for idx, sample_idx in enumerate(sample_indices):
        # 原始光谱
        axes[idx, 0].plot(wavelengths, original[sample_idx], 'b-', linewidth=1.5)
        axes[idx, 0].set_title(f'Original Spectrum (Sample {sample_idx})', fontweight='bold')
        axes[idx, 0].set_xlabel('Wavelength Index', fontweight='bold')
        axes[idx, 0].set_ylabel('Intensity', fontweight='bold')
        axes[idx, 0].grid(True, alpha=0.3)

        # 预处理后的光谱
        axes[idx, 1].plot(wavelengths, processed[sample_idx], 'r-', linewidth=1.5)
        axes[idx, 1].set_title(f'Preprocessed Spectrum (Sample {sample_idx})', fontweight='bold')
        axes[idx, 1].set_xlabel('Wavelength Index', fontweight='bold')
        axes[idx, 1].set_ylabel('Intensity', fontweight='bold')
        axes[idx, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# 训练和验证函数
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=500):
    train_metrics = {"accuracy": [], "f1_score": [], "loss": []}
    val_metrics = {"accuracy": [], "f1_score": [], "loss": []}

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        all_train_preds, all_train_labels = [], []

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
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
        avg_train_loss = train_loss / len(train_loader)

        train_metrics["accuracy"].append(train_acc)
        train_metrics["f1_score"].append(train_f1)
        train_metrics["loss"].append(avg_train_loss)

        # 验证阶段
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)
        val_metrics["accuracy"].append(val_acc)
        val_metrics["f1_score"].append(val_f1)
        val_metrics["loss"].append(val_loss)

        # 学习率调度
        scheduler.step()

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_mobilenet_model.pth')

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            print(f"Train - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            print("-" * 60)

    print("\n" + "=" * 60)
    print("VGG16训练完成!")
    print(f"最终训练指标 - Accuracy: {train_metrics['accuracy'][-1]:.4f}, F1: {train_metrics['f1_score'][-1]:.4f}")
    print(f"最终验证指标 - Accuracy: {val_metrics['accuracy'][-1]:.4f}, F1: {val_metrics['f1_score'][-1]:.4f}")
    print(f"最佳验证准确率: {best_val_acc:.4f}")

    return train_metrics, val_metrics


# 验证函数
def evaluate(model, loader, criterion):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
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
def test_model(model, test_loader, class_names):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)

    # 计算每个类别的准确率
    class_acc = cm.diagonal() / cm.sum(axis=1)

    print("\n" + "=" * 60)
    print("测试结果:")
    print(f"总体准确率: {test_acc:.4f}")
    print(f"加权F1分数: {test_f1:.4f}")
    print("\n各类别准确率:")
    for i, (class_name, acc) in enumerate(zip(class_names, class_acc)):
        print(f"{class_name}: {acc:.4f}")

    return test_acc, test_f1, cm, all_preds, all_labels


# 混淆矩阵绘制
def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=18, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=16, fontweight="bold")
    plt.ylabel("True Label", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


# 训练指标可视化
def plot_training_metrics(train_metrics, val_metrics):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 准确率
    axes[0].plot(train_metrics["accuracy"], label="Train Accuracy", linewidth=2)
    axes[0].plot(val_metrics["accuracy"], label="Validation Accuracy", linewidth=2)
    axes[0].set_title("Accuracy Over Epochs", fontweight="bold")
    axes[0].set_xlabel("Epoch", fontweight="bold")
    axes[0].set_ylabel("Accuracy", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # F1分数
    axes[1].plot(train_metrics["f1_score"], label="Train F1 Score", linewidth=2)
    axes[1].plot(val_metrics["f1_score"], label="Validation F1 Score", linewidth=2)
    axes[1].set_title("F1 Score Over Epochs", fontweight="bold")
    axes[1].set_xlabel("Epoch", fontweight="bold")
    axes[1].set_ylabel("F1 Score", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 损失
    axes[2].plot(train_metrics["loss"], label="Train Loss", linewidth=2)
    axes[2].plot(val_metrics["loss"], label="Validation Loss", linewidth=2)
    axes[2].set_title("Loss Over Epochs", fontweight="bold")
    axes[2].set_xlabel("Epoch", fontweight="bold")
    axes[2].set_ylabel("Loss", fontweight="bold")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# 主程序
if __name__ == "__main__":
    # 数据加载和预处理
    file_path = 'D:/VSCode/fusion/Data/光谱/ALL.xlsx'  # 您的Excel文件路径

    # 预处理参数设置
    preprocessing_params = {
        'apply_sg': True,  # 是否应用SG平滑
        'sg_window': 11,  # SG窗口长度（必须为奇数）
        'sg_polyorder': 2,  # SG多项式阶数
        'sg_deriv': 0,  # SG导数阶数（0=平滑，1=一阶导，2=二阶导）
        'apply_snv': True,  # 是否应用SNV标准化
        'apply_msc': False  # 是否应用MSC校正（与SNV二选一）
    }

    # 加载并预处理数据
    features, labels, label_encoder, scaler, processed_features_before_scaling = load_and_preprocess_data(
        file_path,
        **preprocessing_params
    )

    # 可视化预处理效果（可选）
    # 取原始数据进行对比
    df_original = pd.read_excel(file_path)
    original_features = df_original.iloc[:, 1:].values
    print("\n绘制预处理效果对比图...")
    plot_preprocessing_comparison(
        original_features,
        processed_features_before_scaling,
        sample_indices=[0, 1, 2]  # 显示前3个样本
    )

    # 创建数据集
    dataset = TabularDataset(features, labels)

    # 数据集划分
    train_ratio = 0.7
    val_ratio = 0.1
    test_ratio = 0.2
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # DataLoader
    batch_size = 64  # MobileNet模型较小，可以使用较大的batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 模型初始化
    input_dim = features.shape[1]
    num_classes = len(label_encoder.classes_)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建MobileNet模型
    # width_mult可以调整模型宽度：0.25, 0.5, 0.75, 1.0
    model = MobileNet_1D(
        input_dim=input_dim,
        num_classes=num_classes,
        width_mult=1.0,  # 可以调整为0.5或0.25来获得更轻量的模型
        dropout_rate=0.2
    ).to(device)

    print(f"\nVGG16-1D模型运行设备: {device}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # MobileNet可以使用较大学习率
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-6)

    # 训练模型
    print("\n开始MobileNet模型训练")
    train_metrics, val_metrics = train_and_validate(
        model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=500  # MobileNet收敛较快
    )

    # 可视化训练过程
    plot_training_metrics(train_metrics, val_metrics)

    # 加载最佳模型进行测试
    model.load_state_dict(torch.load('best_mobilenet_model.pth'))

    # 测试模型
    test_acc, test_f1, cm, test_preds, test_labels = test_model(
        model, test_loader, label_encoder.classes_
    )

    # 绘制混淆矩阵
    plot_confusion_matrix(cm, label_encoder.classes_, "Test Set Confusion Matrix")

    print("\n" + "=" * 60)
    print("VGG16模型训练和测试完成!")