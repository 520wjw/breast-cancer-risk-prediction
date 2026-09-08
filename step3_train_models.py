"""
step3_train_models.py
多模型对比训练（8个模型）
网格搜索+交叉验证 + 最优参数训练 + 模型评估
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib

warnings.filterwarnings('ignore')

# 解决中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 模型库
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 评估指标
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix, ConfusionMatrixDisplay
)

# ===================== 1. 加载文件 =====================
print("=" * 50)
print("加载特征选择后的数据...")

# 特征选择后的数据
X_train = np.load('X_train_selected.npy')
X_test = np.load('X_test_selected.npy')

y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

print(f"训练集：{X_train.shape}")
print(f"测试集：{X_test.shape}")

# ===================== 2. 定义模型 + 网格搜索参数 =====================
# 模型与参数网格
models_params = {
    "逻辑回归": {
        "model": LogisticRegression(random_state=42),
        "params": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"]
        }
    },
    "K近邻": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]
        }
    },
    "支持向量机": {
        "model": SVC(random_state=42, probability=True),
        "params": {
            "C": [0.1, 1, 10],
            "gamma": ["scale", "auto"],
            "kernel": ["rbf", "linear"]
        }
    },
    "决策树": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            "max_depth": [3, 5, 7, None],
            "min_samples_split": [2, 5, 10],
            "criterion": ["gini", "entropy"]
        }
    },
    "随机森林": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, None],
            "min_samples_split": [2, 5]
        }
    },
    "朴素贝叶斯": {  # 无超参，跳过网格搜索
        "model": GaussianNB(),
        "params": {}
    },
    "XGBoost": {
        "model": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2]
        }
    },
    "LightGBM": {
        "model": LGBMClassifier(random_state=42, verbose=-1, n_jobs=1, num_threads=1),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "num_leaves": [31, 63, 127]
        }
    }
}

# ===================== 3. 网格搜索 + 交叉验证训练 =====================
print("\n" + "=" * 50)
print("网格搜索 + 5折交叉验证 训练所有模型...")

results = []
cv_results = []  # 交叉验证结果
best_auc = 0
best_model_name = ""
best_model = None
model_fpr_tpr = {}  # 存储ROC曲线数据
best_estimators = {}

for name, mp in models_params.items():
    print(f"\n【{name}】")
    model = mp["model"]
    params = mp["params"]

    # 网格搜索（5折交叉验证）
    if params:
        grid = GridSearchCV(
            estimator=model,
            param_grid=params,
            cv=5,
            scoring="roc_auc",
            n_jobs=2,
            verbose=1
        )
        grid.fit(X_train, y_train)
        best_estimator = grid.best_estimator_
        best_estimators[name] = best_estimator
        print(f"最优参数：{grid.best_params_}")
        print(f"交叉验证最优AUC：{grid.best_score_:.4f}")
    else:
        # 无超参模型直接训练
        best_estimator = model.fit(X_train, y_train)
        best_estimators[name] = best_estimator
        # 5折交叉验证
        cv_auc = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc").mean()
        print(f"交叉验证AUC：{cv_auc:.4f}")

    # 测试集评估
    y_pred = best_estimator.predict(X_test)
    y_pred_proba = best_estimator.predict_proba(X_test)[:, 1]

    # 计算评估指标
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 保存结果
    results.append([name, acc, pre, rec, f1, auc])

    # 保存交叉验证结果（5折）
    cv_aucs = cross_val_score(best_estimator, X_train, y_train, cv=5, scoring="roc_auc")
    cv_results.append([name, cv_aucs.mean(), cv_aucs.std()])

    # 保存ROC曲线数据
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    model_fpr_tpr[name] = (fpr, tpr, auc)

    # 更新最优模型
    if auc > best_auc:
        best_auc = auc
        best_model_name = name
        best_model = best_estimator

    print(f"测试集AUC：{auc:.4f} | 准确率：{acc:.4f}")

# ===================== 4. 结果整理 =====================
# 模型评估结果
df_result = pd.DataFrame(results, columns=["模型", "准确率", "精确率", "召回率", "F1分数", "AUC"])
df_result = df_result.sort_values(by="AUC", ascending=False).reset_index(drop=True)

# 交叉验证结果
df_cv = pd.DataFrame(cv_results, columns=["模型", "CV_AUC均值", "CV_AUC标准差"])
df_cv = df_cv.sort_values(by="CV_AUC均值", ascending=False).reset_index(drop=True)

# 合并结果
df_final = pd.merge(df_result, df_cv, on="模型")

print("\n" + "=" * 60)
print("【模型综合对比排名（含交叉验证）】")
print(df_final.round(4))

print("\n" + "=" * 60)
print(f"最优模型：**{best_model_name}**")
print(f"测试集AUC：{best_auc:.4f}")
print("=" * 60)

# ===================== 5. 保存结果 =====================
# 保存模型对比表（含交叉验证）
df_final.to_csv("model_comparison_with_cv.csv", index=False, encoding="utf-8-sig")

# 保存最优模型及参数
joblib.dump(best_model, "best_model_with_grid.pkl")
# 保存所有模型的最优参数
model_params = {name: (grid.best_params_ if models_params[name]["params"] else {}) for name in models_params.keys()}
pd.DataFrame([{"模型": k, "最优参数": str(v)} for k, v in model_params.items()]).to_csv(
    "best_params.csv", index=False, encoding="utf-8-sig"
)

# ===================== 7. 绘制图表 =====================
print("\n正在生成模型对比图表...")

# 图1：多指标雷达图
def plot_radar_chart(df, top_n=3):
    top_models = df.head(top_n)
    labels = ["准确率", "精确率", "召回率", "F1分数", "AUC"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    for idx, row in top_models.iterrows():
        values = row[labels].tolist()
        values += values[:1]  # 闭合
        ax.plot(angles, values, 'o-', linewidth=1.5, alpha=0.8,
                color=colors[idx], linestyle=linestyles[idx], label=row["模型"])
        ax.fill(angles, values, alpha=0.05, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0.7, 1.0)
    ax.set_title("所有模型多指标对比", fontsize=16, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig("model_radar.png", dpi=300, bbox_inches='tight')
    plt.close()

# 调用函数时指定显示所有8个模型
plot_radar_chart(df_result, top_n=8)


# 图2：交叉验证AUC箱线图
def plot_cv_boxplot(cv_results):
    models = [x[0] for x in cv_results]
    cv_aucs = []
    for name in models:
        cv_aucs.append(cross_val_score(best_estimators[name],
                                       X_train, y_train, cv=5, scoring="roc_auc"))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(cv_aucs, labels=models, patch_artist=True)
    ax.set_title("各模型5折交叉验证AUC分布", fontsize=14)
    ax.set_ylabel("AUC")
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig("model_cv_boxplot.png", dpi=300)
    plt.close()


plot_cv_boxplot(cv_results)

# 图3：ROC曲线合集
plt.figure(figsize=(10, 8))
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
for i, (name, (fpr, tpr, auc_val)) in enumerate(model_fpr_tpr.items()):
    plt.plot(fpr, tpr, color=colors[i], lw=2, label=f"{name} (AUC = {auc_val:.3f})")

plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.xlabel('假阳性率 FPR（误诊率）')
plt.ylabel('真阳性率 TPR（召回率）')
plt.title('所有模型 ROC 曲线对比', fontsize=14)
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("model_roc_all_fixed.png", dpi=300)
plt.close()

# 图4：最优模型混淆矩阵
cm = confusion_matrix(y_test, best_model.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["恶性", "良性"])
disp.plot(cmap=plt.cm.Blues)
plt.title(f"{best_model_name} 混淆矩阵（恶性=0，良性=1）", fontsize=14)
plt.tight_layout()
plt.savefig("best_model_confusion_matrix.png", dpi=300)
plt.close()

# 图5：AUC 对比柱状图
plt.figure(figsize=(10, 5))
plt.bar(df_result["模型"], df_result["AUC"], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'])
plt.xticks(rotation=30, ha='right')
plt.title("各模型 AUC 对比", fontsize=14)
plt.ylim(0.6, 1.0)
plt.tight_layout()
plt.savefig("model_auc_compare.png", dpi=300)
plt.close()

# 图6：准确率对比柱状图
plt.figure(figsize=(10, 5))
plt.bar(df_result["模型"], df_result["准确率"], color='steelblue')
plt.xticks(rotation=30, ha='right')
plt.title("各模型 准确率 对比", fontsize=14)
plt.ylim(0.8, 1.0)
plt.tight_layout()
plt.savefig("model_accuracy_compare.png", dpi=300)
plt.close()

print("\n🎉 网格搜索+交叉验证+可视化 全部完成！")
print("新增文件：")
print("1. model_comparison_with_cv.csv  模型对比表（含交叉验证）")
print("2. best_params.csv               各模型最优参数")
print("3. best_model_with_grid.pkl      最优模型（网格搜索后）")
print("4. model_radar.png               多指标雷达图")
print("5. model_cv_boxplot.png          交叉验证箱线图")
print("6. model_roc_all_fixed.png       ROC曲线")
print("7. best_model_confusion_matrix.png 最优模型混淆矩阵")