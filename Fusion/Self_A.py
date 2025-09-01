import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F

# 定义多头注意力模块
class MultiHeadAttentionForImages(nn.Module):
    def __init__(self, embed_dim, num_heads, patch_size, image_size):
        super(MultiHeadAttentionForImages, self).__init__()
        assert image_size % patch_size == 0, "Image size必须被patch size整除"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2  # 补丁数量
        self.head_dim = embed_dim // num_heads

        # 用于生成 Query, Key, Value 的线性层
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)

        # 最后的线性变换
        self.fc_out = nn.Linear(embed_dim, embed_dim)

        # 补丁嵌入
        self.patch_embedding = nn.Linear(patch_size * patch_size * 3, embed_dim)

        # 可学习的位置编码
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

    def forward(self, x):
        """
        x: 输入图像，形状为 (batch_size, channels, height, width)
        """
        batch_size, channels, height, width = x.shape

        # 1. 切分图像为补丁
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, -1, self.patch_size * self.patch_size * channels)  # (batch_size, num_patches, patch_dim)

        # 2. 补丁嵌入
        patches_embedded = self.patch_embedding(patches)  # (batch_size, num_patches, embed_dim)

        # 3. 添加位置信息
        patches_embedded += self.position_embedding

        # 4. 计算 Query, Key, Value
        Q = self.query_layer(patches_embedded)  # (batch_size, num_patches, embed_dim)
        K = self.key_layer(patches_embedded)    # (batch_size, num_patches, embed_dim)
        V = self.value_layer(patches_embedded)  # (batch_size, num_patches, embed_dim)

        # 5. 多头分割
        Q = Q.view(batch_size, self.num_patches, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, self.num_patches, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, self.num_patches, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 6. 点积注意力
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, num_patches, num_patches)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, num_patches, num_patches)

        # 7. 加权求和
        attention_output = torch.matmul(attention_weights, V)  # (batch_size, num_heads, num_patches, head_dim)

        # 8. 拼接头部输出
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        attention_output = attention_output.view(batch_size, self.num_patches, self.embed_dim)

        # 9. 最后的线性变换
        output = self.fc_out(attention_output)  # (batch_size, num_patches, embed_dim)

        return output, attention_weights


# 定义分类器
class ImageClassifier(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(ImageClassifier, self).__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # 对每个图像的所有补丁取平均值
        x = x.mean(dim=1)  # (batch_size, embed_dim)
        x = self.fc(x)     # (batch_size, num_classes)
        return x


# 定义完整模型：多头注意力 + 分类器
class AttentionBasedImageClassifier(nn.Module):
    def __init__(self, embed_dim, num_heads, patch_size, image_size, num_classes):
        super(AttentionBasedImageClassifier, self).__init__()
        self.attention = MultiHeadAttentionForImages(embed_dim, num_heads, patch_size, image_size)
        self.classifier = ImageClassifier(embed_dim, num_classes)

    def forward(self, x):
        x, _ = self.attention(x)
        x = self.classifier(x)
        return x


# 数据预处理
transform = transforms.Compose([
    transforms.Resize((448, 448)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
])

# 数据路径 (本地文件夹路径)
data_dir = "D:\Data set\RGB_1"

# 加载本地文件夹中的图像数据
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# 打印类别信息
print("Classes:", dataset.classes)
print("Class to Index Mapping:", dataset.class_to_idx)

# 数据加载器
batch_size = 16
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型
embed_dim = 64
num_heads = 4
patch_size = 8
image_size = 448
num_classes = len(dataset.classes)

model = AttentionBasedImageClassifier(embed_dim, num_heads, patch_size, image_size, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
epochs = 30
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
