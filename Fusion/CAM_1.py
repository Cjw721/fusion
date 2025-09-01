from xml.sax.handler import all_features
import torch
import torch.nn as nn
from Concatenate_1 import combined_features_tensor, labels
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x形状要求: [B, C, H, W]
        # 通过自适应池化后, 变为 [B, C, 1, 1]
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        # 将注意力应用到 x 上, 仍维持 [B, C, H, W]
        return self.sigmoid(out).view(x.size(0), x.size(1), 1, 1) * x


  # 1. 打印原始形状 [646, 1226]
print("Original combined_features_tensor shape:", combined_features_tensor.shape)

# 2. 将 [646, 1226] 扩展为 [646, 1226, 1, 1]
features_4d = combined_features_tensor.unsqueeze(-1).unsqueeze(-1)
print("Expanded to 4D shape:", features_4d.shape)
# -> [646, 1226, 1, 1]

# 3. 创建 ChannelAttention 实例
channel_attention = ChannelAttention(in_channels=features_4d.size(1))

# 4. 前向传播，得到 4D 输出
output_4d = channel_attention(features_4d)
print("Output 4D shape:", output_4d.shape)
# -> [646, 1226, 1, 1]

# 5. 如果需要恢复到原来的 2D 形状 [646, 1226]，可以挤掉末尾两个维度
output_2d = output_4d.squeeze(-1).squeeze(-1)
print("Restored to 2D shape:", output_2d.shape)
# -> [646, 1226]

# 最终将此结果视作 CAM 后的特征
CAM_features = output_2d
print("CAM_features type:", type(CAM_features))


############################################    SVM.py    ########################################################
# 转化为numpy数组
X = CAM_features.detach().numpy()
Y = labels
# 划分训练集和测试集   
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# 训练SVM分类器
svm_classifier = SVC(kernel='linear', class_weight='balanced')
svm_classifier.fit(X_train, y_train)

# 预测并评估
y_pred = svm_classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")   
print("\nClassification Report:")
print(classification_report(y_test, y_pred))    

