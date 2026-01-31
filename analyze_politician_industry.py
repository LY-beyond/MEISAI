import pandas as pd
import sys
sys.path.append('task3/src')
import data_processing

# 分析原始数据
data = pd.read_csv('2026_MCM_Problem_C_Data.csv')
# 计算平均评委得分
judge_columns = [col for col in data.columns if 'week' in col.lower() and 'judge' in col.lower() and 'score' in col.lower()]
data['avg_judge_score'] = data[judge_columns].replace(0, pd.NA).mean(axis=1)
# 提取政治家和其他行业选手的评委得分
politician_scores = data[data['celebrity_industry'] == 'Politician']['avg_judge_score']
other_scores = data[data['celebrity_industry'] != 'Politician']['avg_judge_score']
# 转换为数值类型
politician_scores = pd.to_numeric(politician_scores, errors='coerce').dropna()
other_scores = pd.to_numeric(other_scores, errors='coerce').dropna()
# 计算统计信息
print('=== 原始数据分析 ===')
print('政治家行业选手平均评委得分:', politician_scores.mean())
print('其他行业选手平均评委得分:', other_scores.mean())
print('政治家行业选手得分标准差:', politician_scores.std())
print('其他行业选手得分标准差:', other_scores.std())
print('政治家行业选手得分分布:', list(politician_scores.round(2)))
# 查看选手详细信息
print('\n政治家行业选手详细数据:')
politician_data = data[data['celebrity_industry'] == 'Politician']
print(politician_data[['celebrity_name', 'season', 'placement', 'celebrity_age_during_season', 'avg_judge_score']])

# 分析处理后的数据
print('\n=== 处理后数据分析 ===')
processed_data = data_processing.load_and_preprocess_data()
print('处理后数据中政治家行业选手数量:', len(processed_data[processed_data['industry_simplified'] == 'Politician']))
politician_processed = processed_data[processed_data['industry_simplified'] == 'Politician']
print('政治家行业选手平均职业舞者经验:', politician_processed['pro_experience'].mean())
print('其他行业选手平均职业舞者经验:', processed_data[processed_data['industry_simplified'] != 'Politician']['pro_experience'].mean())
print('\n政治行业与评分的相关性:')
print(politician_processed[['avg_judge_score', 'predicted_fan_votes', 'celebrity_age_during_season', 'pro_experience']].describe())