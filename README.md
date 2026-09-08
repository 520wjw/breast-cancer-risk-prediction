# Breast Cancer Risk Prediction（乳腺癌风险预测模型研究）

基于 sklearn 乳腺癌数据集的机器学习分类研究：异常值处理、归一化、特征选择、**8 种模型对比**、网格搜索调优与可解释性分析。

## 技术栈

Python · scikit-learn · XGBoost · LightGBM · pandas · matplotlib

## 处理流程

| 脚本 | 说明 |
|---|---|
| `step1_precess.py` | 数据预处理：IQR 异常值检测与过滤、归一化、数据集划分 |
| `step2_feature_selection.py` | 特征选择 |
| `step3_train_models.py` | 8 模型对比（LR / KNN / SVM / Decision Tree / Random Forest / NB / XGBoost / LightGBM），网格搜索 + 交叉验证调优，输出精度 / AUC / 混淆矩阵评估 |
| `step4_explain.py` | 模型可解释性分析 |

## 结果可视化

- `model_accuracy_compare.png` / `model_auc_compare.png`：模型精度与 AUC 对比
- `model_cv_boxplot.png` / `model_radar.png`：交叉验证箱线图与综合能力雷达图
- `model_comparison_with_cv.csv` / `best_params.csv`：对比结果与最优参数
- `anova_f_score.png` / `feature_selection_*.csv`：特征选择分析

## 说明

数据集来自 sklearn 内置 `breast_cancer` 数据集，无需额外下载。
