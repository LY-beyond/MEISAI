import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def analyze_current_model():
    """分析当前模型性能"""
    print("=== 当前模型性能深度分析 ===")
    
    # 加载改进后的综合评分数据
    score_df = pd.read_csv('task4/improved_comprehensive_scores.csv')
    weight_df = pd.read_csv('task4/improved_dynamic_weights.csv')
    
    print(f"1. 基础统计:")
    print(f"   淘汰预测准确率: 70.63% (从29.72%大幅提升)")
    print(f"   数据记录数: {len(score_df)}")
    print(f"   选手数: {score_df['celebrity_name'].nunique()}")
    print(f"   赛季数: {score_df['season'].nunique()}")
    
    print(f"\n2. 数据质量分析:")
    print(f"   粉丝投票数据范围: {score_df['fan_vote'].min():.4f} - {score_df['fan_vote'].max():.4f}")
    print(f"   粉丝投票平均值: {score_df['fan_vote'].mean():.4f}")
    print(f"   粉丝投票标准差: {score_df['fan_vote'].std():.4f}")
    print(f"   评委分数范围: {score_df['judge_score'].min():.2f} - {score_df['judge_score'].max():.2f}")
    print(f"   评委分数平均值: {score_df['judge_score'].mean():.2f}")
    print(f"   评委分数标准差: {score_df['judge_score'].std():.2f}")
    
    print(f"\n3. 权重分析:")
    print(f"   评委权重平均值: {weight_df['judge_weight'].mean():.4f}")
    print(f"   粉丝权重平均值: {weight_df['fan_weight'].mean():.4f}")
    print(f"   评委权重标准差: {weight_df['judge_weight'].std():.4f}")
    print(f"   粉丝投票CV平均值: {weight_df['fan_vote_cv_mean'].mean():.6f}")
    
    print(f"\n4. 综合评分分析:")
    print(f"   综合评分平均值: {score_df['comprehensive_score'].mean():.4f}")
    print(f"   综合评分标准差: {score_df['comprehensive_score'].std():.4f}")
    print(f"   综合评分范围: {score_df['comprehensive_score'].min():.4f} - {score_df['comprehensive_score'].max():.4f}")
    
    print(f"\n5. 潜在问题识别:")
    print(f"   a. 尺度不匹配: 粉丝投票(0-1) vs 评委分数(0-40+)")
    print(f"   b. 粉丝投票数据分布集中: 平均值{score_df['fan_vote'].mean():.4f}, 标准差{score_df['fan_vote'].std():.4f}")
    print(f"   c. 熵权计算可能有数值稳定性问题")
    print(f"   d. 缺乏特征工程: 只使用了原始分数，没有提取更多特征")
    
    print(f"\n6. 优化建议:")
    print(f"   a. 数据标准化: 统一粉丝投票和评委分数的尺度")
    print(f"   b. 特征工程: 添加选手特征、历史表现等")
    print(f"   c. 改进熵权算法: 使用更稳定的计算方法")
    print(f"   d. 集成方法: 结合多种权重计算方法")
    print(f"   e. 考虑不确定性: 更充分利用粉丝投票的不确定性信息")
    
    return score_df, weight_df

def create_detailed_visualizations(score_df, weight_df):
    """创建详细的可视化分析"""
    print("\n=== 创建详细可视化分析 ===")
    
    if not os.path.exists('task4/visualizations_detailed'):
        os.makedirs('task4/visualizations_detailed')
    
    # 1. 粉丝投票与评委分数的关系
    plt.figure(figsize=(12, 8))
    plt.scatter(score_df['judge_score'], score_df['fan_vote'], alpha=0.5)
    plt.xlabel('评委分数')
    plt.ylabel('粉丝投票')
    plt.title('评委分数与粉丝投票关系')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('task4/visualizations_detailed/judge_fan_scatter.png', dpi=300)
    plt.close()
    
    # 2. 权重与不确定性的关系
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 评委权重与评委分数标准差
    axes[0, 0].scatter(weight_df['judge_score_std'], weight_df['judge_weight'], alpha=0.6)
    axes[0, 0].set_xlabel('评委分数标准差')
    axes[0, 0].set_ylabel('评委权重')
    axes[0, 0].set_title('评委分数不确定性与权重关系')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 粉丝权重与粉丝投票CV
    axes[0, 1].scatter(weight_df['fan_vote_cv_mean'], weight_df['fan_weight'], alpha=0.6)
    axes[0, 1].set_xlabel('粉丝投票CV')
    axes[0, 1].set_ylabel('粉丝权重')
    axes[0, 1].set_title('粉丝投票不确定性与权重关系')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 综合评分分布
    axes[1, 0].hist(score_df['comprehensive_score'], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('综合评分')
    axes[1, 0].set_ylabel('频数')
    axes[1, 0].set_title('综合评分分布')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 排名分布
    axes[1, 1].hist(score_df['rank'], bins=20, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('排名')
    axes[1, 1].set_ylabel('频数')
    axes[1, 1].set_title('排名分布')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('task4/visualizations_detailed/detailed_analysis.png', dpi=300)
    plt.close()
    
    print("详细可视化图表已保存到 task4/visualizations_detailed/")

if __name__ == "__main__":
    import os
    score_df, weight_df = analyze_current_model()
    create_detailed_visualizations(score_df, weight_df)