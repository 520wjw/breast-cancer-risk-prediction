"""
step1_preprocess.py
数据预处理模块：异常值处理 + 数据归一化 + 数据集划分
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

# ===================== 1. 加载原始数据集 =====================
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target  # 添加标签：0=恶性，1=良性

print("=" * 50)
print("原始数据形状：", df.shape)

# ===================== 2. 基础数据检查 =====================
print("\n" + "=" * 50)
print("【数据基础检查】")
print("缺失值数量：", df.isnull().sum().sum())
print("数据类型：\n", df.dtypes)
print("标签分布：\n", df['target'].value_counts())

# ===================== 3. 异常值检测与过滤 =====================
print("\n" + "=" * 50)
print("【异常值检测与处理】")

def remove_outliers_iqr(df, feature_cols):
    """
    使用IQR方法删除异常值
    :param df: 原始数据
    :param feature_cols: 特征列名列表
    :return: 去除异常值后的数据
    """
    df_clean = df.copy()
    for col in feature_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # 保留在范围内的数据
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]

    return df_clean


# 分离特征列与标签列
feature_cols = df.columns.drop('target')
df_no_outlier = remove_outliers_iqr(df, feature_cols)

print("去除异常值前：", df.shape)
print("去除异常值后：", df_no_outlier.shape)
print("异常值已过滤完成！")

# ===================== 4. 数据标准化 =====================
print("\n" + "=" * 50)
print("【数据标准化】")

# 划分特征 X 和标签 y
X = df_no_outlier[feature_cols]
y = df_no_outlier['target']

# 标准化（均值0，方差1）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 转回DataFrame方便后续使用
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
X_scaled_df.reset_index(drop=True, inplace=True)
y_reset = y.reset_index(drop=True)

# 合并标准化后的特征+标签
df_scaled = pd.concat([X_scaled_df, y_reset], axis=1)

print("归一化完成，数据已统一到均值=0，标准差=1")

# ===================== 5. 划分训练集 / 测试集（7:3） =====================
print("\n" + "=" * 50)
print("【划分训练集与测试集】")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print("训练集形状：", X_train.shape)
print("测试集形状：", X_test.shape)
print("训练集标签分布：\n", pd.Series(y_train).value_counts())
print("测试集标签分布：\n", pd.Series(y_test).value_counts())

# ===================== 6. 保存预处理结果 =====================
print("\n" + "=" * 50)
print("【保存预处理后数据】")

# 保存完整标准化数据
df_scaled.to_csv('data_scaled.csv', index=False, encoding='utf-8')

# 保存训练集、测试集
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

# 保存列名
with open('feature_names.csv', 'w', encoding='utf-8') as f:
    f.write('\n'.join(feature_cols))  # 直接写入特征名，每行一个

print("✅ 预处理全部完成！")
print("已保存文件：")
print("1. data_scaled.csv    标准化完整数据")
print("2. X_train.npy        训练集特征")
print("3. X_test.npy         测试集特征")
print("4. y_train.npy        训练集标签")
print("5. y_test.npy         测试集标签")
print("6. feature_names.csv  特征列名")
print("=" * 50)