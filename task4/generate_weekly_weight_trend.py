import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def generate_weekly_weight_trend():
    """生成每周动态变化趋势图"""
    print("=== 生成每周动态变化趋势图 ===")
    
    # 加载修复后的数据
    df = pd.read_csv('ultimate_90_percent_scores_fixed.csv')
    
    # 过滤掉NaN值的权重数据
    df_weights = df.dropna(subset=['judge_weight', 'fan_weight'])
    
    # 按赛季和周计算平均权重
    weekly_avg_weights = df_weights.groupby(['season', 'week'])[['judge_weight', 'fan_weight']].mean().reset_index()
    
    print(f"有效数据记录数: {len(df_weights)}")
    print(f"每周平均权重记录数: {len(weekly_avg_weights)}")
    
    # 创建可视化目录
    viz_dir = 'visualizations_ultimate'
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    # 生成每周动态变化趋势图
    plt.figure(figsize=(16, 10))
    
    # 显示所有赛季
    seasons_to_show = weekly_avg_weights['season'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(seasons_to_show)))
    
    for i, season in enumerate(seasons_to_show):
        season_data = weekly_avg_weights[weekly_avg_weights['season'] == season]
        
        plt.plot(season_data['week'], season_data['judge_weight'], 
                marker='o', color=colors[i], label=f'S{season} 评委权重', alpha=0.7)
        plt.plot(season_data['week'], season_data['fan_weight'], 
                marker='s', color=colors[i], label=f'S{season} 粉丝权重', alpha=0.3, linestyle='--')
    
    plt.xlabel('周数', fontsize=12)
    plt.ylabel('权重', fontsize=12)
    plt.title('每周动态权重变化趋势（终极模型）', fontsize=14)
    
    # 添加图例
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(viz_dir, 'weekly_weight_trend.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"图表已保存到: {output_path}")
    
    # 同时生成所有赛季的聚合趋势图
    plt.figure(figsize=(12, 8))
    
    # 按周计算所有赛季的平均权重
    overall_trend = weekly_avg_weights.groupby('week')[['judge_weight', 'fan_weight']].mean().reset_index()
    
    plt.plot(overall_trend['week'], overall_trend['judge_weight'], 
            marker='o', color='blue', label='平均评委权重', linewidth=2)
    plt.plot(overall_trend['week'], overall_trend['fan_weight'], 
            marker='s', color='red', label='平均粉丝权重', linewidth=2)
    
    plt.xlabel('周数', fontsize=12)
    plt.ylabel('权重', fontsize=12)
    plt.title('每周平均权重变化趋势（所有赛季聚合）', fontsize=14)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path2 = os.path.join(viz_dir, 'overall_weight_trend.png')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"聚合趋势图已保存到: {output_path2}")
    
    # 输出统计信息
    print("\\n=== 统计信息 ===")
    print(f"赛季数量: {df['season'].nunique()}")
    print(f"周数范围: {df['week'].min()} - {df['week'].max()}")
    print("\\n=== 权重统计 ===")
    print("评委权重:")
    print(f"  平均值: {df_weights['judge_weight'].mean():.4f}")
    print(f"  标准差: {df_weights['judge_weight'].std():.4f}")
    print(f"  范围: {df_weights['judge_weight'].min():.4f} - {df_weights['judge_weight'].max():.4f}")
    print("粉丝权重:")
    print(f"  平均值: {df_weights['fan_weight'].mean():.4f}")
    print(f"  标准差: {df_weights['fan_weight'].std():.4f}")
    print(f"  范围: {df_weights['fan_weight'].min():.4f} - {df_weights['fan_weight'].max():.4f}")
    
    return weekly_avg_weights, overall_trend

if __name__ == "__main__":
    import numpy as np
    weekly_avg_weights, overall_trend = generate_weekly_weight_trend()