import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
base_dir = 'd:/MEISAI'
rank_path = os.path.join(base_dir, 'task1', 'dwts_rank_regular_processed.csv')
percentage_path = os.path.join(base_dir, 'task1', 'dwts_percentage_regular_processed.csv')
bottom_path = os.path.join(base_dir, 'task1', 'dwts_rank_bottom_two_processed.csv')

print("正在读取数据...")
df1 = pd.read_csv(rank_path, encoding='cp1252')
df2 = pd.read_csv(percentage_path, encoding='cp1252') 
df3 = pd.read_csv(bottom_path, encoding='cp1252')

# 合并
df = pd.concat([df1, df2, df3], ignore_index=True)

print(f'总数据条数: {len(df)}')
print(f'不同舞伴数量: {df["ballroom_partner"].nunique()}')

print('\n舞伴分布（前20名）:')
partner_counts = df['ballroom_partner'].value_counts().head(20)
print(partner_counts)

# 计算平均评委得分
judge_cols = [col for col in df.columns if 'judge' in col.lower() and 'score' in col.lower()]
df['avg_judge_score'] = df[judge_cols].replace(0, np.nan).mean(axis=1)

# 分组统计
partner_stats = df.groupby('ballroom_partner').agg({
    'avg_judge_score': 'mean',
    'placement': 'mean',  # 排名（数值越小越好）
    'celebrity_name': 'count'  # 配对次数
}).rename(columns={'celebrity_name': 'pair_count', 'placement': 'avg_placement'})

# 筛选出至少配对5次的舞伴
partner_stats_filtered = partner_stats[partner_stats['pair_count'] >= 5].sort_values('avg_placement')
print('\n舞伴表现统计（至少配对5次）:')
print(partner_stats_filtered.head(20))

# 分析舞伴经验（职业舞者决赛/获胜次数）
pro_dancer_experience = df.groupby('ballroom_partner')['placement'].apply(
    lambda x: sum((x <= 3) & (x > 0))
).reset_index()
pro_dancer_experience.columns = ['ballroom_partner', 'pro_experience']

# 合并经验数据
partner_stats_filtered = partner_stats_filtered.merge(pro_dancer_experience, on='ballroom_partner', how='left')

print('\n舞伴经验与表现相关性分析:')
print(partner_stats_filtered[['avg_judge_score', 'avg_placement', 'pro_experience']].corr())

# 可视化1：舞伴表现排名
plt.figure(figsize=(12, 8))
top_partners = partner_stats_filtered.sort_values('avg_placement').head(15)
plt.barh(range(len(top_partners)), top_partners['avg_placement'])
plt.yticks(range(len(top_partners)), top_partners.index)
plt.xlabel('平均排名（数值越小越好）')
plt.title('最佳舞伴排名（至少配对5次）')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('d:/MEISAI/partner_performance_ranking.png', dpi=300, bbox_inches='tight')
print('\n已保存可视化：partner_performance_ranking.png')

# 可视化2：舞伴经验与表现散点图
plt.figure(figsize=(10, 6))
plt.scatter(partner_stats_filtered['pro_experience'], partner_stats_filtered['avg_placement'])
plt.xlabel('职业舞者经验（历史决赛/获胜次数）')
plt.ylabel('平均排名（数值越小越好）')
plt.title('舞伴经验与表现关系')
for idx, row in partner_stats_filtered.iterrows():
    if row['pair_count'] >= 8:  # 只标记配对次数较多的舞伴
        plt.annotate(idx, (row['pro_experience'], row['avg_placement']), fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('d:/MEISAI/partner_experience_vs_performance.png', dpi=300, bbox_inches='tight')
print('已保存可视化：partner_experience_vs_performance.png')

# 可视化3：舞伴配对次数分布
plt.figure(figsize=(12, 6))
all_partner_counts = df['ballroom_partner'].value_counts()
plt.hist(all_partner_counts, bins=range(1, all_partner_counts.max()+2), edgecolor='black', alpha=0.7)
plt.xlabel('配对次数')
plt.ylabel('舞伴数量')
plt.title('舞伴配对次数分布')
plt.axvline(x=5, color='red', linestyle='--', label='分析阈值（5次）')
plt.legend()
plt.tight_layout()
plt.savefig('d:/MEISAI/partner_pairing_distribution.png', dpi=300, bbox_inches='tight')
print('已保存可视化：partner_pairing_distribution.png')

# 保存详细分析结果
output_path = os.path.join(base_dir, 'partner_analysis_summary.txt')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('《与星共舞》舞伴因素分析报告\n')
    f.write('=' * 50 + '\n\n')
    
    f.write('1. 数据概览\n')
    f.write(f'   总数据条数: {len(df)}\n')
    f.write(f'   不同舞伴数量: {df["ballroom_partner"].nunique()}\n\n')
    
    f.write('2. 舞伴配对次数分布（前20名）\n')
    f.write(partner_counts.to_string() + '\n\n')
    
    f.write('3. 舞伴表现统计（至少配对5次）\n')
    f.write(partner_stats_filtered.to_string() + '\n\n')
    
    f.write('4. 相关性分析\n')
    f.write('舞伴经验与表现相关性矩阵:\n')
    f.write(partner_stats_filtered[['avg_judge_score', 'avg_placement', 'pro_experience']].corr().to_string() + '\n\n')
    
    f.write('5. 关键发现\n')
    best_partners = partner_stats_filtered.sort_values('avg_placement').head(5)
    f.write('   最佳舞伴（平均排名最低）:\n')
    for idx, row in best_partners.iterrows():
        f.write(f'   - {idx}: 平均排名 {row["avg_placement"]:.2f}, 配对次数 {row["pair_count"]}\n')
    
    worst_partners = partner_stats_filtered.sort_values('avg_placement', ascending=False).head(5)
    f.write('\n   最差舞伴（平均排名最高）:\n')
    for idx, row in worst_partners.iterrows():
        f.write(f'   - {idx}: 平均排名 {row["avg_placement"]:.2f}, 配对次数 {row["pair_count"]}\n')

print(f'\n分析完成！详细报告已保存至: {output_path}')