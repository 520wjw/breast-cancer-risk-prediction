# Breast Cancer Risk Prediction（乳腺癌风险预测模型研究）

基于 sklearn 内置乳腺癌数据集的完整机器学习分类研究：**数据预处理 → 特征选择 → 8 种模型对比调优 → SHAP 可解释性分析**，全流程含可视化输出。

## 特性

- 🧹 **规范预处理**：IQR 异常值检测与过滤 + 标准化 + 训练/测试划分
- 🔬 **双方法特征选择**：随机森林特征重要性 + ANOVA F 检验取交集，bootstrap（5 次）稳定性验证
- ⚖️ **8 模型对比**：LR / KNN / SVM / Decision Tree / Random Forest / Naive Bayes / XGBoost / LightGBM
- 🎯 **网格搜索 + 交叉验证**：GridSearchCV 调参，5 折 CV 稳定性评估
- 📊 **SHAP 可解释性**：全局特征重要性、单样本力场图、特征交互分析

## 实验结果（测试集，来自 `model_comparison_with_cv.csv`）

| 模型 | 准确率 | 精确率 | 召回率 | F1 | AUC | CV-AUC 均值 ± 标准差 |
|---|---|---|---|---|---|---|
| **LightGBM** | **0.9405** | 0.9390 | **1.0000** | **0.9686** | 0.9397 | 0.9112 ± 0.095 |
| **K近邻** | **0.9405** | **0.9500** | 0.9870 | 0.9682 | 0.8952 | 0.8387 ± 0.145 |
| 支持向量机 | 0.9286 | 0.9277 | 1.0000 | 0.9625 | **0.9425** | 0.9422 ± 0.055 |
| 逻辑回归 | 0.9286 | 0.9494 | 0.9740 | 0.9615 | 0.9406 | **0.9496** ± 0.049 |
| XGBoost | 0.9167 | 0.9167 | 1.0000 | 0.9565 | 0.8849 | 0.8538 ± 0.197 |
| 随机森林 | 0.9167 | 0.9375 | 0.9740 | 0.9554 | 0.8571 | 0.9587 ± 0.029 |
| 决策树 | 0.9167 | 0.9487 | 0.9610 | 0.9548 | 0.7486 | 0.8092 ± 0.126 |
| 朴素贝叶斯 | 0.9048 | 0.9859 | 0.9091 | 0.9459 | 0.9295 | 0.9311 ± 0.094 |

**要点**：LightGBM / KNN 取得最高准确率（94.05%）；SVM 测试集 AUC 最高（0.9425）；逻辑回归交叉验证 AUC 最稳定（0.9496 ± 0.049）。

## 处理流程

| 脚本 | 内容 |
|---|---|
| `step1_precess.py` | 数据加载、缺失值检查、IQR 异常值过滤、StandardScaler 归一化、train_test_split |
| `step2_feature_selection.py` | 随机森林特征重要性 + ANOVA F 检验 → 交集特征集；bootstrap 稳定性验证；特征选择可视化 |
| `step3_train_models.py` | 8 模型网格搜索（GridSearchCV）+ 5 折交叉验证 + 最优参数训练 + 精度/AUC/混淆矩阵评估 |
| `step4_explain.py` | SHAP 全局/局部解释、特征影响量化、Top-N 特征交互分析 |

## 结果可视化

| 图片 | 说明 |
|---|---|
| `model_accuracy_compare.png` | 8 模型准确率对比 |
| `model_auc_compare.png` | 8 模型 AUC 对比 |
| `model_cv_boxplot.png` | 交叉验证得分箱线图（稳定性） |
| `model_radar.png` | 多指标综合雷达图 |
| `best_model_confusion_matrix.png` | 最优模型混淆矩阵 |
| `anova_f_score.png` | ANOVA F 检验得分 |
| `rf_feature_importance.png` | 随机森林特征重要性 |
| `shap_summary_beeswarm.png` | SHAP 蜂群图 |
| `shap_feature_importance_bar.png` | SHAP 特征重要性条形图 |
| `shap_force_single_sample.png` | 单样本 SHAP 力场图 |
| `shap_interaction_*.png` | 特征交互效应图 |

## 快速开始

```bash
pip install scikit-learn xgboost lightgbm pandas numpy matplotlib shap joblib

# 依次运行
python step1_precess.py
python step2_feature_selection.py
python step3_train_models.py
python step4_explain.py
```

> 数据集来自 sklearn 内置 `load_breast_cancer()`（569 样本 × 30 特征），**无需额外下载**，复现零门槛。

## 输出产物

- `data_scaled.csv` / `*.npy`：预处理与特征选择后的数据
- `model_comparison_with_cv.csv` / `best_params.csv`：对比结果与最优参数
- `final_selected_features.csv` / `feature_selection_*.csv`：特征选择明细
- `shap_*.csv` / `shap_analysis_conclusion.txt`：SHAP 量化结果与结论
