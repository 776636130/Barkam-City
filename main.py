import os
import pandas as pd
import matplotlib.pyplot as plt
import umap
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from xgboost import XGBClassifier, plot_importance
import shap

# 设置字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
df = pd.read_csv('maerk13.csv', encoding='utf-8')
print(df.head())  # 显示数据框的前几行

# 提取所有指标，排除第一列（序号）和最后一列（ZHD）
viz_columns = df.columns[1:-1]  # 获取除第一列和最后一列外的所有列名

# 检查缺失值
print("缺失值统计：")
print(df[viz_columns].isnull().sum())

# 提取高维特征用于降维
scaled_features = df[viz_columns].dropna().values  # 确保特征数据没有缺失值

# 定义 UMAP 降维函数
def apply_umap(features: np.array, random_state: int = 7, metric: str = 'euclidean', neighbor_num: int = 15):
    reducer = umap.UMAP(random_state=random_state, n_neighbors=neighbor_num, metric=metric, n_jobs=1)
    embedding = reducer.fit_transform(features)
    return reducer, embedding

# 只保留 'euclidean' 作为度量
metrics = ['euclidean']

# 创建一个2行1列的子图布局
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))

# 遍历不同的度量
for i, metric in enumerate(metrics):
    # UMAP 降维
    _, embedding = apply_umap(scaled_features, neighbor_num=18, metric=metric)

    # 使用 HDBSCAN 进行聚类
    hdb = HDBSCAN(min_cluster_size=150)
    labels = hdb.fit_predict(embedding)

    # 将聚类标签添加到数据框中的特定列
    df[f'HDBSCAN_label_{metric}'] = labels  # 将聚类标签添加到 df 的特定列

    ax = axes  # 当前子图
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis', alpha=0.5)
    ax.set_title(f'使用度量 {metric} 的 HDBSCAN 聚类结果')
    ax.set_xlabel('UMAP 特征 1')
    ax.set_ylabel('UMAP 特征 2')
    ax.grid(True)

    # 计算每个聚类的中心点并标注
    unique_labels = np.unique(labels)  # 获取所有唯一的聚类标签
    # 统计每个聚类的数量
    label_counts = {label: np.sum(labels == label) for label in unique_labels if label != -1}
    print(f"使用 {metric} 度量的聚类数量统计：", label_counts)  # 打印聚类数量

    # 按数量降序排列，选择前5个
    top_labels = sorted(label_counts, key=label_counts.get, reverse=True)[:5]

    # 仅标注前5个聚类
    for label in top_labels:
        if label in label_counts and label_counts[label] > 0:  # 检查聚类是否有数据
            cluster_points = embedding[labels == label]  # 获取该聚类的所有点
            center_point = cluster_points.mean(axis=0)  # 计算中心点
            ax.text(center_point[0], center_point[1], str(label), fontsize=10, ha='center', va='center', color='red')

# 添加颜色条
plt.colorbar(scatter, ax=ax, label='聚类标签')
plt.tight_layout()
plt.show()

# 遍历指定的聚类标签
for label in top_labels:
    if label in [0, 1, 2, 3, 4]:
        print(f'Processing importance and SHAP plots for label: {label} with metric: {metric}')

    print(f'processing labels {label}')
    df['label'] = 0
    df.loc[df[f'HDBSCAN_label_{metric}'] == label, 'label'] = 1

    # 数据拆分
    X_train, X_test, y_train, y_test = train_test_split(
        df[viz_columns].values,
        df['label'],
        stratify=df['label'],
        test_size=0.2
    )

    print(np.unique(y_train, return_counts=True)[-1])  # 打印训练集每个类的数量
    train_df = pd.DataFrame(X_train, columns=viz_columns)
    test_df = pd.DataFrame(X_test, columns=viz_columns)

    # 计算类权重
    classes_weights = class_weight.compute_sample_weight(
        class_weight='balanced',
        y=y_train
    )

    # 训练XGBoost模型
    tree = XGBClassifier(n_estimators=1288, max_depth=6, objective='binary:logistic', sample_weight=classes_weights)
    tree.fit(train_df, y_train)

    preds = tree.predict(X_test)
    print(f"\nLabel {label}: Classification Report:")
    print(classification_report(y_test, preds))

    # 绘制特征重要性
    plot_importance(tree)
    plt.title(f"Label {label}: xgboost.plot_importance(model)")
    plt.show()

    # SHAP值解释
    explainer = shap.Explainer(tree, train_df)
    shap_values = explainer(train_df)

    # 设置字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    shap.plots.bar(shap_values[1])




