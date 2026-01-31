# 2026 MCM Problem C 项目结构

## 项目概述

本项目针对《Dancing with the Stars》（DWTS）节目数据进行全面分析，主要解决以下问题：

1. 粉丝投票预测模型构建
2. 投票方法对比分析
3. 影响因素分析（职业舞者表现及名人特征）
4. 公平投票系统设计

## 目录结构

### 根目录文件

- `2026_MCM_Problem_C_Data.csv`: 原始数据文件
- `requirements.txt`: Python依赖库列表
- `data_preprocessing.py`: 数据预处理脚本
- `advanced_data_preprocessing.py`: 高级数据预处理脚本
- `fan_vote_model.py`: 粉丝投票预测模型
- `create_visualizations.py`: 可视化创建脚本
- `analyze_data.py`: 数据分析脚本
- `data_validation.py`: 数据验证脚本
- `fair_voting_system.py`: 公平投票系统设计
- `pro_dancer_analysis.py`: 职业舞者分析
- `executive_memo.md`: 执行备忘录（英文）
- `executive_memo_zh.md`: 执行备忘录（中文）

### task1 目录 - 粉丝投票预测

- `data.py`: Task1主数据处理脚本
- `data copy.py`: 数据处理备份脚本
- `fan_vote_predictions_enhanced.csv`: 增强版粉丝投票预测结果
- `fan_vote_weights_analysis.csv`: 粉丝投票权重分析
- `voting_methods_comparison.csv`: 投票方法对比结果
- `voting_methods_comparison_report.txt`: 投票方法对比报告
- `dwts_percentage_regular_processed.csv`: 百分比法处理后数据
- `dwts_rank_regular_processed.csv`: 排名法处理后数据
- `dwts_rank_bottom_two_processed.csv`: 排名法+评委选择处理后数据
- `dwts_uncertainty_metrics.csv`: 不确定性指标
- `uncertainty_analysis_summary.csv`: 不确定性分析总结
- `controversial_cases_analysis.csv`: 争议案例分析
- `judge_choice_simulation.csv`: 评委选择模拟结果

#### task1/visualizations 目录 - Task1可视化结果

- `comparison/`: 方法对比可视化
- `uncertainty/`: 不确定性可视化
- `additional/`: 补充可视化

### task3 目录 - 影响因素分析（XGBoost+SHAP模型）

- `influence_analysis.py`: 影响因素分析主脚本
- `data_processing.py`: 数据处理模块
- `model_training.py`: 模型训练模块
- `shap_analysis.py`: SHAP分析模块
- `visualizations/`: 可视化结果

### visualizations 目录 - 全局可视化

- 主项目可视化结果（如桑基图）

### 桑吉图目录 - 桑基图可视化

- 粉丝投票与排名关系的桑基图

## 任务流程

### Task 1: 粉丝投票预测

1. 数据预处理与清理
2. 建立粉丝投票预测模型
3. 验证模型一致性与不确定性
4. 可视化结果展示

### Task 2: 投票方法对比

1. 实现两种投票组合方法（排名法和百分比法）
2. 对比分析两种方法的结果差异
3. 评估方法对粉丝投票的偏向性
4. 分析争议案例的处理结果

### Task 3: 影响因素分析（XGBoost+SHAP模型）

1. 数据准备与特征工程
2. 构建XGBoost回归模型（评委得分和粉丝投票）
3. SHAP值分析与解释
4. 可视化特征重要性与交互效应

### Task 4: 公平投票系统设计

1. 设计新的公平投票系统
2. 评估系统的公平性和效果
3. 提供实施建议

## 技术栈

- Python 3.x
- Pandas, NumPy: 数据处理
- Scikit-learn: 机器学习
- XGBoost: 梯度提升模型
- SHAP: 解释性分析
- Matplotlib, Seaborn, Plotly: 可视化
- Scipy: 科学计算

## 运行说明

1. 安装依赖：`pip install -r requirements.txt`
2. 运行各任务脚本
3. 查看生成的CSV和可视化文件
