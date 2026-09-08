"""
step2_feature_selection.py
特征选择：双方法 + 取交集 + 可视化 + 稳定性验证
方法1：随机森林特征重要性
方法2：ANOVA F 检验（方差分析）
最终取两种方法的特征交集作为最优特征集
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils import resample
import warnings
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')

# 解决中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 配置参数（可灵活调整） =====================
CONFIG = {
    "selection_strategy": "percent",  # percent: 按百分比筛选 | fixed: 固定数量
    "top_percent": 0.3,               # 百分比模式：筛选前30%特征
    "top_k": 9,                       # 固定数量模式：筛选前9个特征
    "n_bootstrap": 5,                 # 特征稳定性验证的重采样次数
    "random_state": 42
}

# ===================== 1. 加载预处理后的数据 =====================
logging.info("=" * 50)
logging.info("加载预处理完成的数据...")

try:
    # 加载上一步保存的文件
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    with open('feature_names.csv', 'r', encoding='utf-8') as f:
        feature_names = [line.strip() for line in f if line.strip()]  # 去空行、去空格

    logging.info(f"训练集特征形状：{X_train.shape}")
    logging.info(f"总特征数量：{len(feature_names)}")

    # 数据校验
    if X_train.shape[1] != len(feature_names):
        raise ValueError(f"特征数量与列名数量不匹配！特征维度={X_train.shape[1]}, 列名数量={len(feature_names)}")

except FileNotFoundError as e:
    logging.error(f"文件加载失败：{e}")
    raise
except Exception as e:
    logging.error(f"数据加载异常：{e}")
    raise

# ===================== 辅助函数：特征稳定性验证 =====================
def validate_feature_stability(X, y, feature_names, n_bootstrap=5, top_k=9):
    """
    通过重采样验证特征重要性的稳定性
    """
    feature_selection_count = {name: 0 for name in feature_names}

    for i in range(n_bootstrap):
        X_resampled, y_resampled = resample(X, y, random_state=CONFIG['random_state'] + i)
        rf = RandomForestClassifier(n_estimators=100, random_state=CONFIG['random_state'])
        rf.fit(X_resampled, y_resampled)
        importance = rf.feature_importances_
        top_idx = np.argsort(importance)[::-1][:top_k]
        top_features = [feature_names[idx] for idx in top_idx]

        for feat in top_features:
            feature_selection_count[feat] += 1

    # 计算特征被选中的频率（越高越稳定）
    stability_score = {k: v / n_bootstrap for k, v in feature_selection_count.items()}
    return stability_score

# ===================== 2. 方法1：随机森林特征重要性 =====================
# ===================== 2. 方法1：随机森林特征重要性 =====================
logging.info("\n" + "=" * 50)
logging.info("【方法1：随机森林 特征重要性筛选】")

# 训练随机森林并计算特征重要性
rf = RandomForestClassifier(n_estimators=100, random_state=CONFIG['random_state'])
rf.fit(X_train, y_train)
importance = rf.feature_importances_

# 确定筛选数量
if CONFIG["selection_strategy"] == "percent":
    top_n = int(len(feature_names) * CONFIG["top_percent"])
else:
    top_n = CONFIG["top_k"]
top_n = max(1, top_n)  # 至少保留1个特征

# 筛选Top N特征
rf_idx = np.argsort(importance)[::-1][:top_n]
rf_selected = [feature_names[i] for i in rf_idx]

# 特征稳定性验证
stability_score = validate_feature_stability(X_train, y_train, feature_names,
                                            n_bootstrap=CONFIG['n_bootstrap'], top_k=top_n)
rf_selected_stable = [feat for feat in rf_selected if stability_score[feat] >= 0.8]  # 筛选稳定特征（频率≥80%）

logging.info(f"随机森林原始筛选特征数：{len(rf_selected)}")
logging.info(f"随机森林稳定特征数（频率≥80%）：{len(rf_selected_stable)}")
logging.info(f"随机森林选中稳定特征：\n {rf_selected_stable}")

# 可视化随机森林特征重要性
plt.figure(figsize=(12, 6))
feat_importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance,
    "stability": [stability_score[f] for f in feature_names]
}).sort_values(by="importance", ascending=False).head(top_n * 2)

plt.barh(feat_importance_df["feature"][::-1], feat_importance_df["importance"][::-1], color='lightcoral')
plt.xlabel("特征重要性")
plt.title(f"随机森林 Top {top_n*2} 特征重要性", fontsize=12)
plt.tight_layout()
plt.savefig("rf_feature_importance.png", dpi=300, bbox_inches='tight')
plt.close()
logging.info("✅ 已保存：rf_feature_importance.png（随机森林特征重要性图）")

# ===================== 3. 方法2：ANOVA F 检验（过滤式） =====================
logging.info("\n" + "=" * 50)
logging.info("【方法2：ANOVA F 检验 特征筛选】")

# 选择TOP K个特征（与随机森林保持一致数量）
selector = SelectKBest(score_func=f_classif, k=top_n)
selector.fit(X_train, y_train)
anova_scores = selector.scores_
anova_pvalues = selector.pvalues_

# 筛选特征
anova_idx = selector.get_support(indices=True)
anova_selected = [feature_names[i] for i in anova_idx]

# 过滤p值>0.05的特征（统计显著性）
anova_selected_significant = [feat for idx, feat in enumerate(anova_selected)
                              if anova_pvalues[idx] < 0.05]

logging.info(f"ANOVA原始筛选特征数：{len(anova_selected)}")
logging.info(f"ANOVA显著特征数（p<0.05）：{len(anova_selected_significant)}")
logging.info(f"ANOVA选中显著特征：\n {anova_selected_significant}")

# 可视化ANOVA F检验结果
plt.figure(figsize=(12, 6))
anova_df = pd.DataFrame({
    "feature": feature_names,
    "f_score": anova_scores,
    "p_value": anova_pvalues
}).sort_values(by="f_score", ascending=False).head(top_n * 2)

plt.barh(anova_df["feature"][::-1], anova_df["f_score"][::-1], color='lightskyblue')
plt.xlabel("F值（越大越显著）")
plt.title(f"ANOVA F检验 Top {top_n*2} 特征", fontsize=12)
plt.tight_layout()
plt.savefig("anova_f_score.png", dpi=300, bbox_inches='tight')
plt.close()
logging.info("✅ 已保存：anova_f_score.png（ANOVA F值分布图）")

# ===================== 4. 取两种方法的特征交集 =====================
logging.info("\n" + "=" * 50)
logging.info("【取两种方法的特征交集 → 最终特征集】")

# 取稳定+显著特征的交集
final_features = list(set(rf_selected_stable) & set(anova_selected_significant))

# 兜底：如果交集为空，退回到随机森林稳定特征
if len(final_features) == 0:
    logging.warning("⚠️ 两种方法无交集特征，退回到随机森林稳定特征")
    final_features = rf_selected_stable[:top_n//2]  # 取一半数量

final_features.sort()  # 排序保持稳定

logging.info(f"\n✅ 最终筛选完成！共保留 {len(final_features)} 个特征")
logging.info(f"最终最优特征：\n {final_features}")

# 输出特征筛选详情
selection_detail = pd.DataFrame({
    "feature": feature_names,
    "rf_importance": importance,
    "anova_f_score": anova_scores,
    "anova_p_value": anova_pvalues,
    "stability_score": [stability_score[f] for f in feature_names],
    "is_final": [1 if f in final_features else 0 for f in feature_names]
})
selection_detail.to_csv("feature_selection_detail.csv", index=False, encoding='utf-8')
logging.info("✅ 已保存：feature_selection_detail.csv（特征筛选详情）")

# ===================== 5. 生成筛选后的新数据（供建模用） =====================
logging.info("\n" + "=" * 50)
logging.info("生成筛选后的训练/测试集...")

# 获取索引（防KeyError）
final_idx = []
for f in final_features:
    if f in feature_names:
        final_idx.append(feature_names.index(f))
    else:
        logging.warning(f"⚠️ 特征{f}不存在，已跳过")

# 筛选特征
X_train_selected = X_train[:, final_idx]
X_test_selected = X_test[:, final_idx]

logging.info(f"筛选后训练集形状：{X_train_selected.shape}")
logging.info(f"筛选后测试集形状：{X_test_selected.shape}")

# ===================== 6. 保存结果（给step3建模使用） =====================
logging.info("\n" + "=" * 50)
logging.info("保存特征选择结果...")

# 保存最终特征列表
pd.DataFrame(final_features, columns=['final_features']).to_csv(
    'final_selected_features.csv', index=False, encoding='utf-8'
)

# 保存筛选后的数据集
np.save('X_train_selected.npy', X_train_selected)
np.save('X_test_selected.npy', X_test_selected)

# 保存筛选配置
pd.DataFrame([CONFIG]).to_csv("feature_selection_config.csv", index=False, encoding='utf-8')

logging.info("✅ 特征选择全部完成！")
logging.info("已保存：")
logging.info("1. final_selected_features.csv  最终特征列表")
logging.info("2. X_train_selected.npy        筛选后训练集")
logging.info("3. X_test_selected.npy         筛选后测试集")
logging.info("4. rf_feature_importance.png   随机森林特征重要性图")
logging.info("5. anova_f_score.png           ANOVA F值分布图")
logging.info("6. feature_selection_detail.csv 特征筛选详情")
logging.info("7. feature_selection_config.csv 筛选配置")
logging.info("=" * 50)