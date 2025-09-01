import torch
import torch.nn as nn
from CAM_1 import CAM_features

# ====================================
# 2. 定义 Spatial Attention (SAM)
# ====================================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # input channels=2 (concat(avg_out, max_out)), output channels=1
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        # 对通道维 (dim=1) 做 max 和 mean
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # -> [B, 1, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)    # -> [B, 1, H, W]

        # 拼接 => [B, 2, H, W]
        x_cat = torch.cat([avg_out, max_out], dim=1)

        # 卷积后用 sigmoid => 空间注意力图 [B, 1, H, W]
        attention_map = self.sigmoid(self.conv(x_cat))

        # 返回乘以原输入 x 的结果
        return attention_map * x

# ====================================
# 3. 使用 SAM
# ====================================

# 3.1 原始形状 [646, 1226]
print("Original:", CAM_features.shape)

# 3.2 扩展维度 => [646, 1226, 1, 1]
#     视其为 batch_size=646, channels=1226, height=1, width=1
features_4d = CAM_features.unsqueeze(-1).unsqueeze(-1)
print("Expanded to 4D:", features_4d.shape)

# 3.3 构建并应用 SAM
sam = SpatialAttention(kernel_size=7)
sam_out = sam(features_4d)
print("After SAM:", sam_out.shape)  # [646, 1226, 1, 1]

# 3.4 恢复到 2D => [646, 1226]
final_features = sam_out.squeeze(-1).squeeze(-1)
print("Restored to 2D:", final_features.shape)
SAM_features = final_features
print("SAM_features type:", type(SAM_features))
