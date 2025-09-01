import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
# 文件夹路径
image_features_dir = r"D:\Data set\concatenate\RGB_Fea"
spectral_features_dir = r"D:\Data set\concatenate\NIR_Pre"

# # 真实类别标签
# class_labels = ['Harm', 'Health', 'Locusts', 'Mildew']
# 读取特征文件函数
def load_features(image_dir, spectral_dir):
    image_features = []
    spectral_features = []
    labels = []

    class_folders = sorted(os.listdir(image_dir))

    for label, class_folder in enumerate(class_folders):
        image_class_dir = os.path.join(image_dir, class_folder)
        spectral_class_dir = os.path.join(spectral_dir, class_folder)

        image_files = sorted([f for f in os.listdir(image_class_dir) if f.endswith('.csv')])
        spectral_files = sorted([f for f in os.listdir(spectral_class_dir) if f.endswith('.csv')])

        for img_file, spec_file in zip(image_files, spectral_files):
            img_df = pd.read_csv(os.path.join(image_class_dir, img_file), header=None)
            spec_df = pd.read_csv(os.path.join(spectral_class_dir, spec_file), header=None)

            image_features.append(img_df.iloc[0].values)
            spectral_features.append(spec_df.iloc[0].values)
            labels.append(label)

    return np.array(image_features), np.array(spectral_features), np.array(labels)

# Connect the features
def concatenate_features(image_features, spectral_features):
    return np.concatenate((image_features, spectral_features), axis=1)

# 加载特征
image_features, spectral_features, labels = load_features(image_features_dir, spectral_features_dir)

# 连接特征
combined_features = concatenate_features(image_features, spectral_features)

# 打印结果
print("Image Features Shape:", image_features.shape)
print("Spectral Features Shape:", spectral_features.shape)
print("Combined Features Shape:", combined_features.shape)
print("Labels Shape:", labels.shape)
print(type(combined_features))
# 将combined_features从numpy.ndarray转换为torch.Tensor
combined_features_tensor = torch.from_numpy(combined_features).float()
print(combined_features_tensor.shape)
print(type(combined_features_tensor))



############################################    SVM.py    ########################################################
# 转化为numpy数组
X = combined_features_tensor.detach().numpy()
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


