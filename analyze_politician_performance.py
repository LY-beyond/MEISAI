import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('task3/src')
import data_processing

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 分析原始数据
data = pd.read_csv('2026_MCM_Problem_C_Data.csv')
# 计算平均评委得分
judge_columns = [col for col in data.columns if 'week' in col.lower() and 'judge' in col.lower() and 'score' in col.lower()]
data['avg_judge_score'] = data[judge_columns].replace(0, pd.NA).mean(axis=1)

# 提取政治家行业选手数据
politician_data = data[data['celebrity_industry'] == 'Politician']
# 提取其他行业选手数据（为了公平比较，只包含与政治家同年龄组的选手）
other_data = data[data['celebrity_industry'] != 'Politician']
age_50_plus = other_data[other_data['celebrity_age_during_season'] >= 50]

print("=== 政治家行业选手详细表现 ===")
print(politician_data[['celebrity_name', 'season', 'placement', 'celebrity_age_during_season', 'avg_judge_score']])

print("\n=== 50岁以上其他行业选手平均得分 ===")
print(f"平均得分: {age_50_plus['avg_judge_score'].mean():.2f}")
print(f"得分标准差: {age_50_plus['avg_judge_score'].std():.2f}")

# 查看原始数据的列名
print("\n=== 原始数据列名 ===")
print(list(data.columns))

# 分析处理后的数据
processed_data = data_processing.load_and_preprocess_data()
politician_processed = processed_data[processed_data['industry_simplified'] == 'Politician']

print("\n=== 处理后数据中政治家选手的特征 ===")
print(politician_processed[['celebrity_name', 'pro_experience', 'predicted_fan_votes', 'is_american']])

# 可视化分析
plt.figure(figsize=(10, 6))

# 1. 年龄与平均得分的关系
plt.subplot(1, 2, 1)
sns.scatterplot(x='celebrity_age_during_season', y='avg_judge_score', data=other_data, alpha=0.5, label='其他行业')
sns.scatterplot(x='celebrity_age_during_season', y='avg_judge_score', data=politician_data, color='red', s=100, label='政治家')
plt.xlabel('年龄')
plt.ylabel('平均评委得分')
plt.title('年龄与平均得分的关系')
plt.legend()

# 2. 行业与平均得分的箱线图
plt.subplot(1, 2, 2)
# 只显示主要行业
top_industries = data['celebrity_industry'].value_counts().head(5).index
filtered_data = data[data['celebrity_industry'].isin(top_industries) | (data['celebrity_industry'] == 'Politician')]
sns.boxplot(x='celebrity_industry', y='avg_judge_score', data=filtered_data)
plt.xticks(rotation=45, ha='right')
plt.xlabel('行业')
plt.ylabel('平均评委得分')
plt.title('行业与平均得分的关系')

plt.tight_layout()
plt.savefig('politician_performance_analysis.png', dpi=300, bbox_inches='tight')
print("\n已保存可视化图表到 politician_performance_analysis.png")

# 计算相关系数
print("\n=== 相关系数分析 ===")
correlation_matrix = other_data[['celebrity_age_during_season', 'avg_judge_score']].corr()
print("年龄与得分的相关系数 (其他行业):")
print(correlation_matrix)

# 查看政治家选手与舞蹈经验的关系
print("\n=== 政治家选手与舞蹈经验的关系 ===")
dancer_data = data[data['celebrity_industry'] == 'Dancer']
print(f"舞蹈家选手平均得分: {dancer_data['avg_judge_score'].mean():.2f}")
print(f"舞蹈家选手平均年龄: {dancer_data['celebrity_age_during_season'].mean():.2f}")