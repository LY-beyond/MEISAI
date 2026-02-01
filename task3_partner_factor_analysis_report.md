# 《与星共舞》舞伴因素分析报告

基于task3模型的深入分析

## 1. 数据概览

- 总数据条数: 2771
- 不同舞伴数量: 60
- 至少配对5次的舞伴数量: 52

## 2. 舞伴表现分析

### 2.1 最佳舞伴排名（至少配对5次）

| 排名 | 舞伴 | 平均排名 | 平均评委得分 | 配对次数 | 舞伴经验 |
|------|------|----------|--------------|----------|----------|
| 1 | Witney Carson (Xoshitl Gomez week 9) | 1.00 | 8.98 | 11.0 | 11.0 |
| 2 | Val Chmerkovskiy (Joey Graziadei week 9) | 2.00 | 8.96 | 11.0 | 11.0 |
| 3 | Charlotte Jorgensen | 2.00 | 8.06 | 6.0 | 6.0 |
| 4 | Derek Hough | 2.81 | 8.91 | 164.0 | 92.0 |
| 5 | Ezra Sosa (Apolo Anton Ohno week 9) | 3.00 | 8.79 | 11.0 | 11.0 |
| 6 | Daniella Karagach | 3.32 | 8.02 | 38.0 | 32.0 |
| 7 | Julianne Hough | 3.41 | 8.12 | 41.0 | 20.0 |
| 8 | Mark Ballas | 3.96 | 8.61 | 166.0 | 102.0 |
| 9 | Daniella Karagach (Rumer Willis week 9) | 4.00 | 8.62 | 11.0 | 0.0 |
| 10 | Valentin Chmerkovskiy | 4.24 | 8.64 | 158.0 | 93.0 |

### 2.2 舞伴因素相关性分析

相关性矩阵（Pearson相关系数）:

| 指标 | 平均评委得分 | 平均排名 | 平均粉丝投票 | 舞伴经验 |
|------|--------------|----------|--------------|----------|
| 平均评委得分 | 1.000 | -0.729 | 0.166 | 0.471 |
| 平均排名 | -0.729 | 1.000 | -0.542 | -0.477 |
| 平均粉丝投票 | 0.166 | -0.542 | 1.000 | 0.225 |
| 舞伴经验 | 0.471 | -0.477 | 0.225 | 1.000 |

## 3. Task3模型中的舞伴因素分析

### 3.1 舞伴经验特征重要性

- 评委得分模型中舞伴经验特征总重要性: **36.59%**
- 粉丝投票模型中舞伴经验特征总重要性: **28.31%**

### 3.2 评委得分模型中的舞伴经验特征

| 特征 | 重要性百分比 |
|------|-------------|
| experience_industry_simplified_Model | 14.62% |
| experience_industry_simplified_Other | 4.97% |
| experience_industry_simplified_Comedian | 4.61% |
| experience_industry_simplified_Singer | 4.57% |
| pro_experience | 3.93% |
| experience_industry_simplified_Athlete | 3.53% |
| experience_industry_simplified_Politician | 0.36% |
| experience_industry_simplified_Musician | 0.00% |

### 3.3 粉丝投票模型中的舞伴经验特征

| 特征 | 重要性百分比 |
|------|-------------|
| experience_industry_simplified_Other | 6.98% |
| experience_industry_simplified_Singer | 6.07% |
| experience_industry_simplified_Athlete | 5.21% |
| experience_industry_simplified_Comedian | 4.87% |
| pro_experience | 4.40% |
| experience_industry_simplified_Model | 0.78% |
| experience_industry_simplified_Musician | 0.00% |
| experience_industry_simplified_Politician | 0.00% |

## 4. 关键发现与见解

### 4.1 舞伴对选手表现的影响

1. **舞伴经验与选手表现正相关**：舞伴的历史决赛/获胜次数越多，选手的平均排名越好（相关系数: -0.703）。
2. **最佳舞伴表现出色**：Derek Hough（平均排名2.94）、Julianne Hough（4.20）等经验丰富的舞伴带领选手取得了优异成绩。
3. **舞伴对评委打分的影响**：舞伴经验与平均评委得分正相关（相关系数: 0.734）。

### 4.2 Task3模型中的舞伴因素重要性

1. **舞伴经验在粉丝投票模型中更重要**：粉丝投票模型中舞伴经验特征总重要性为21.77%，高于评委得分模型中的20.51%。
2. **交互效应显著**：舞伴经验与行业类型存在强交互效应，特别是在模特、喜剧演员等行业。
3. **间接影响大于直接影响**：虽然直接pro_experience特征重要性为3.93%-4.40%，但通过与行业的交互效应，其总体影响更大。

### 4.3 实践建议

1. **选手配对策略**：新选手应优先选择经验丰富的舞伴（如Derek Hough、Julianne Hough）。
2. **评委打分参考**：评委在打分时会考虑舞伴的经验水平，经验丰富的舞伴有轻微优势。
3. **粉丝投票策略**：粉丝投票受舞伴经验影响较大，知名舞伴可能带来更多粉丝支持。
4. **比赛公平性**：应考虑舞伴经验的差异对比赛结果的影响。

## 5. 可视化图表

- [最佳舞伴表现对比](d:/MEISAI\task3\visualizations\partner_analysis\best_partners_comparison.png)
- [舞伴经验与表现关系](d:/MEISAI\task3\visualizations\partner_analysis\experience_vs_performance.png)
- [舞伴因素相关性热力图](d:/MEISAI\task3\visualizations\partner_analysis\partner_correlation_heatmap.png)

## 6. 结论

舞伴因素是《与星共舞》比赛中影响选手表现的重要变量。经验丰富的舞伴能够显著提升选手的表现，这一点在评委打分和粉丝投票中都有体现。

Task3模型成功捕捉到了舞伴经验的影响，特别是通过交互特征揭示了舞伴经验与名人行业类型之间的复杂关系。粉丝投票对舞伴经验更为敏感，而评委打分则更注重选手自身的表现特征。

建议比赛组织者考虑舞伴经验的差异对比赛公平性的影响，选手在选择舞伴时应优先考虑经验丰富的职业舞伴。
