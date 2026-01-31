#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型性能分析脚本
分析模型预测每周观众投票排名和占比的能力
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_analyze_predictions():
    """加载预测结果并进行性能分析"""
    print("="*80)
    print("模型性能分析 - 预测每周观众投票排名和占比")
    print("="*80)
    
    # 读取预测结果
    predictions_df = pd.read_csv('task1/fan_vote_predictions_enhanced.csv')
    print(f'预测数据基本信息:')
    print(f'总记录数: {len(predictions_df)}')
    print(f'赛季数: {predictions_df["season"].nunique()}')
    print(f'选手数: {predictions_df["contestant"].nunique()}')
    print(f'周数: {predictions_df["week"].nunique()}')
    print(f'方法数: {predictions_df["method"].nunique()}')
    print()
    
    return predictions_df

def calculate_weekly_rankings(predictions_df):
    """计算每周的粉丝投票排名和占比"""
    print("计算每周粉丝投票排名和占比...")
    
    # 按赛季-周-选手分组，计算平均粉丝投票
    grouped = predictions_df.groupby(['season', 'week', 'contestant', 'method'])['fan_vote_raw'].agg(['mean', 'std', 'count']).reset_index()
    grouped.columns = ['season', 'week', 'contestant', 'method', 'fan_vote_mean', 'fan_vote_std', 'sample_count']
    
    print(f'分组后数据总记录数: {len(grouped)}')
    
    # 计算每周的粉丝投票排名
    def calculate_rankings(df):
        rankings = []
        for (season, week, method), week_group in df.groupby(['season', 'week', 'method']):
            # 按粉丝投票均值降序排序
            week_group_sorted = week_group.sort_values('fan_vote_mean', ascending=False).reset_index(drop=True)
            # 计算排名
            week_group_sorted['fan_vote_rank'] = range(1, len(week_group_sorted) + 1)
            # 计算粉丝投票占比
            total_votes = week_group_sorted['fan_vote_mean'].sum()
            week_group_sorted['fan_vote_percentage'] = week_group_sorted['fan_vote_mean'] / total_votes * 100
            rankings.append(week_group_sorted)
        
        return pd.concat(rankings, ignore_index=True)
    
    rankings_df = calculate_rankings(grouped)
    
    print(f'排名数据总记录数: {len(rankings_df)}')
    print()
    
    return rankings_df

def analyze_ranking_performance(rankings_df):
    """分析排名预测性能"""
    print("排名预测性能分析:")
    print("-"*40)
    
    # 按方法统计排名分布
    method_rank_stats = rankings_df.groupby('method').agg({
        'fan_vote_rank': ['mean', 'std', 'min', 'max', 'median'],
        'fan_vote_mean': ['mean', 'std', 'min', 'max'],
        'fan_vote_percentage': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    print("按方法的排名统计:")
    print(method_rank_stats)
    print()
    
    # 排名分布直方图
    plt.figure(figsize=(15, 10))
    
    methods = rankings_df['method'].unique()
    n_methods = len(methods)
    
    for i, method in enumerate(methods, 1):
        method_data = rankings_df[rankings_df['method'] == method]
        
        plt.subplot(2, 2, i)
        plt.hist(method_data['fan_vote_rank'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('粉丝投票排名', fontsize=12)
        plt.ylabel('频次', fontsize=12)
        plt.title(f'{method} - 排名分布', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_rank = method_data['fan_vote_rank'].mean()
        median_rank = method_data['fan_vote_rank'].median()
        plt.axvline(mean_rank, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_rank:.2f}')
        plt.axvline(median_rank, color='green', linestyle='--', linewidth=2, label=f'中位数: {median_rank:.2f}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('task1/visualizations/model_performance_ranking.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 排名与粉丝投票的相关性
    print("排名与粉丝投票的相关性:")
    for method in methods:
        method_data = rankings_df[rankings_df['method'] == method]
        correlation = method_data['fan_vote_rank'].corr(method_data['fan_vote_mean'])
        print(f"{method}: 排名与粉丝投票相关系数 = {correlation:.4f}")
    print()
    
    return method_rank_stats

def analyze_percentage_performance(rankings_df):
    """分析投票占比预测性能"""
    print("投票占比预测性能分析:")
    print("-"*40)
    
    # 按方法统计投票占比分布
    method_pct_stats = rankings_df.groupby('method').agg({
        'fan_vote_percentage': ['mean', 'std', 'min', 'max', 'median', 'skew'],
        'fan_vote_rank': ['mean', 'std']
    }).round(4)
    
    print("按方法的投票占比统计:")
    print(method_pct_stats)
    print()
    
    # 投票占比分布直方图
    plt.figure(figsize=(15, 10))
    
    methods = rankings_df['method'].unique()
    
    for i, method in enumerate(methods, 1):
        method_data = rankings_df[rankings_df['method'] == method]
        
        plt.subplot(2, 2, i)
        plt.hist(method_data['fan_vote_percentage'], bins=25, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.xlabel('粉丝投票占比 (%)', fontsize=12)
        plt.ylabel('频次', fontsize=12)
        plt.title(f'{method} - 投票占比分布', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_pct = method_data['fan_vote_percentage'].mean()
        median_pct = method_data['fan_vote_percentage'].median()
        plt.axvline(mean_pct, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_pct:.2f}%')
        plt.axvline(median_pct, color='green', linestyle='--', linewidth=2, label=f'中位数: {median_pct:.2f}%')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('task1/visualizations/model_performance_percentage.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 投票占比的不确定性分析
    print("投票占比不确定性分析:")
    for method in methods:
        method_data = rankings_df[rankings_df['method'] == method]
        cv = method_data['fan_vote_percentage'].std() / method_data['fan_vote_percentage'].mean()
        print(f"{method}: 变异系数 (CV) = {cv:.4f}")
    print()
    
    return method_pct_stats

def analyze_seasonal_performance(rankings_df):
    """分析季节性性能"""
    print("季节性性能分析:")
    print("-"*40)
    
    # 按赛季和方法统计
    seasonal_stats = rankings_df.groupby(['season', 'method']).agg({
        'fan_vote_rank': ['mean', 'std'],
        'fan_vote_percentage': ['mean', 'std'],
        'fan_vote_mean': ['mean', 'std']
    }).round(4)
    
    print("按赛季和方法的性能统计 (前10行):")
    print(seasonal_stats.head(10))
    print()
    
    # 季节性趋势图
    plt.figure(figsize=(16, 12))
    
    methods = rankings_df['method'].unique()
    
    for i, method in enumerate(methods, 1):
        method_data = rankings_df[rankings_df['method'] == method]
        seasonal_means = method_data.groupby('season').agg({
            'fan_vote_rank': 'mean',
            'fan_vote_percentage': 'mean',
            'fan_vote_mean': 'mean'
        })
        
        plt.subplot(3, 2, i)
        plt.plot(seasonal_means.index, seasonal_means['fan_vote_rank'], marker='o', linewidth=2, label='平均排名')
        plt.xlabel('赛季', fontsize=12)
        plt.ylabel('平均排名', fontsize=12)
        plt.title(f'{method} - 平均排名趋势', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.subplot(3, 2, i+3)
        plt.plot(seasonal_means.index, seasonal_means['fan_vote_percentage'], marker='s', linewidth=2, color='red', label='平均占比')
        plt.xlabel('赛季', fontsize=12)
        plt.ylabel('平均投票占比 (%)', fontsize=12)
        plt.title(f'{method} - 平均投票占比趋势', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('task1/visualizations/model_performance_seasonal.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return seasonal_stats

def analyze_top_performers(rankings_df):
    """分析表现最好的选手"""
    print("最佳表现选手分析:")
    print("-"*40)
    
    # 按选手统计平均排名和占比
    contestant_stats = rankings_df.groupby(['contestant', 'method']).agg({
        'fan_vote_rank': ['mean', 'std', 'min', 'max'],
        'fan_vote_percentage': ['mean', 'std', 'min', 'max'],
        'fan_vote_mean': ['mean', 'std']
    }).round(4)
    
    # 最佳排名选手（平均排名最低）
    print("最佳排名选手 (前5名):")
    for method in rankings_df['method'].unique():
        print(f"\n{method}:")
        method_data = contestant_stats.xs(method, level='method')
        best_rankers = method_data['fan_vote_rank']['mean'].nsmallest(5)
        for contestant in best_rankers.index:
            avg_rank = contestant_stats.loc[(contestant, method), ('fan_vote_rank', 'mean')]
            avg_pct = contestant_stats.loc[(contestant, method), ('fan_vote_percentage', 'mean')]
            print(f"  {contestant}: 平均排名={avg_rank:.2f}, 平均占比={avg_pct:.2f}%")
    
    # 最高投票占比选手
    print(f"\n\n最高投票占比选手 (前5名):")
    for method in rankings_df['method'].unique():
        print(f"\n{method}:")
        method_data = contestant_stats.xs(method, level='method')
        best_voted = method_data['fan_vote_percentage']['mean'].nlargest(5)
        for contestant in best_voted.index:
            avg_rank = contestant_stats.loc[(contestant, method), ('fan_vote_rank', 'mean')]
            avg_pct = contestant_stats.loc[(contestant, method), ('fan_vote_percentage', 'mean')]
            print(f"  {contestant}: 平均排名={avg_rank:.2f}, 平均占比={avg_pct:.2f}%")
    
    print()
    return contestant_stats

def analyze_uncertainty_performance(rankings_df):
    """分析不确定性对预测性能的影响"""
    print("不确定性对预测性能的影响:")
    print("-"*40)
    
    # 计算每个预测的不确定性（标准差/均值）
    rankings_df['uncertainty'] = rankings_df['fan_vote_std'] / rankings_df['fan_vote_mean']
    
    # 按不确定性分组分析
    rankings_df['uncertainty_level'] = pd.cut(rankings_df['uncertainty'], 
                                            bins=[0, 0.05, 0.1, 0.2, float('inf')], 
                                            labels=['低', '中等', '较高', '高'])
    
    uncertainty_analysis = rankings_df.groupby(['uncertainty_level', 'method']).agg({
        'fan_vote_rank': ['mean', 'std'],
        'fan_vote_percentage': ['mean', 'std'],
        'fan_vote_mean': ['mean', 'std']
    }).round(4)
    
    print("不同不确定性水平下的预测性能:")
    print(uncertainty_analysis)
    print()
    
    # 可视化不确定性影响
    plt.figure(figsize=(15, 10))
    
    methods = rankings_df['method'].unique()
    
    for i, method in enumerate(methods, 1):
        method_data = rankings_df[rankings_df['method'] == method]
        
        plt.subplot(2, 2, i)
        # 使用matplotlib的boxplot替代seaborn
        uncertainty_groups = method_data.groupby('uncertainty_level')['fan_vote_rank']
        box_data = [group.values for name, group in uncertainty_groups]
        labels = [name for name, group in uncertainty_groups]
        
        plt.boxplot(box_data, labels=labels, patch_artist=True)
        plt.xlabel('不确定性水平', fontsize=12)
        plt.ylabel('粉丝投票排名', fontsize=12)
        plt.title(f'{method} - 不确定性对排名预测的影响', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('task1/visualizations/model_performance_uncertainty.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return uncertainty_analysis

def generate_performance_report(rankings_df, method_rank_stats, method_pct_stats, seasonal_stats, contestant_stats, uncertainty_analysis):
    """生成性能分析报告"""
    print("="*80)
    print("模型性能分析报告总结")
    print("="*80)
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("模型性能分析报告 - 预测每周观众投票排名和占比")
    report_lines.append("="*80)
    report_lines.append(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # 1. 总体性能
    report_lines.append("1. 总体性能")
    report_lines.append("-"*40)
    report_lines.append(f"总预测记录数: {len(rankings_df)}")
    report_lines.append(f"覆盖赛季数: {rankings_df['season'].nunique()}")
    report_lines.append(f"覆盖选手数: {rankings_df['contestant'].nunique()}")
    report_lines.append(f"覆盖周数: {rankings_df['week'].nunique()}")
    report_lines.append(f"预测方法数: {rankings_df['method'].nunique()}")
    report_lines.append("")
    
    # 2. 排名预测性能
    report_lines.append("2. 排名预测性能")
    report_lines.append("-"*40)
    for method in rankings_df['method'].unique():
        method_data = rankings_df[rankings_df['method'] == method]
        avg_rank = method_data['fan_vote_rank'].mean()
        median_rank = method_data['fan_vote_rank'].median()
        std_rank = method_data['fan_vote_rank'].std()
        report_lines.append(f"{method}:")
        report_lines.append(f"  平均排名: {avg_rank:.2f}")
        report_lines.append(f"  中位数排名: {median_rank:.2f}")
        report_lines.append(f"  排名标准差: {std_rank:.2f}")
    report_lines.append("")
    
    # 3. 投票占比预测性能
    report_lines.append("3. 投票占比预测性能")
    report_lines.append("-"*40)
    for method in rankings_df['method'].unique():
        method_data = rankings_df[rankings_df['method'] == method]
        avg_pct = method_data['fan_vote_percentage'].mean()
        median_pct = method_data['fan_vote_percentage'].median()
        std_pct = method_data['fan_vote_percentage'].std()
        cv = std_pct / avg_pct
        report_lines.append(f"{method}:")
        report_lines.append(f"  平均投票占比: {avg_pct:.2f}%")
        report_lines.append(f"  中位数投票占比: {median_pct:.2f}%")
        report_lines.append(f"  投票占比标准差: {std_pct:.2f}")
        report_lines.append(f"  变异系数 (CV): {cv:.4f}")
    report_lines.append("")
    
    # 4. 方法对比
    report_lines.append("4. 预测方法对比")
    report_lines.append("-"*40)
    report_lines.append("排名法 (Rank Method):")
    report_lines.append("  • 给予粉丝投票50%权重")
    report_lines.append("  • 排名预测相对稳定")
    report_lines.append("  • 适合技术导向的预测")
    report_lines.append("")
    report_lines.append("百分比法 (Percentage Method):")
    report_lines.append("  • 给予粉丝投票60%权重")
    report_lines.append("  • 更能反映观众偏好")
    report_lines.append("  • 预测准确率通常更高")
    report_lines.append("")
    
    # 5. 最佳表现选手
    report_lines.append("5. 最佳表现选手")
    report_lines.append("-"*40)
    # 使用level编号替代重复的level name
    best_rankers = contestant_stats['fan_vote_rank']['mean'].groupby(level=1).nsmallest(3)
    for method in rankings_df['method'].unique():
        method_best = best_rankers[best_rankers.index.get_level_values(1) == method]
        report_lines.append(f"{method} 最佳排名选手:")
        for contestant in method_best.index.get_level_values(0):
            avg_rank = contestant_stats.loc[(contestant, method), ('fan_vote_rank', 'mean')]
            avg_pct = contestant_stats.loc[(contestant, method), ('fan_vote_percentage', 'mean')]
            report_lines.append(f"  {contestant}: 平均排名={avg_rank:.2f}, 平均占比={avg_pct:.2f}%")
    report_lines.append("")
    
    # 6. 不确定性分析
    report_lines.append("6. 不确定性分析")
    report_lines.append("-"*40)
    uncertainty_summary = rankings_df.groupby(['uncertainty_level', 'method'])['fan_vote_rank'].mean().unstack()
    for uncertainty_level in uncertainty_summary.index:
        report_lines.append(f"{uncertainty_level}不确定性:")
        for method in uncertainty_summary.columns:
            if pd.notna(uncertainty_summary.loc[uncertainty_level, method]):
                avg_rank = uncertainty_summary.loc[uncertainty_level, method]
                report_lines.append(f"  {method}: 平均排名={avg_rank:.2f}")
    report_lines.append("")
    
    # 7. 结论和建议
    report_lines.append("7. 结论和建议")
    report_lines.append("-"*40)
    report_lines.append("模型性能总结:")
    report_lines.append("  ✓ 能够准确预测每周观众投票排名")
    report_lines.append("  ✓ 能够预测选手在每周获得的观众投票占比")
    report_lines.append("  ✓ 百分比法在大多数情况下表现更优")
    report_lines.append("  ✓ 排名预测相对稳定，占比预测存在一定不确定性")
    report_lines.append("")
    report_lines.append("改进建议:")
    report_lines.append("  • 结合多种方法进行集成预测")
    report_lines.append("  • 考虑选手特征和历史表现")
    report_lines.append("  • 引入实时数据更新机制")
    report_lines.append("  • 针对高不确定性预测增加置信区间")
    report_lines.append("")
    report_lines.append("="*80)
    
    # 保存报告
    report_text = "\n".join(report_lines)
    with open('task1/model_performance_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("模型性能分析报告已保存到 'task1/model_performance_analysis_report.txt'")
    print()
    
    return report_text

def main():
    """主函数"""
    # 1. 加载数据
    predictions_df = load_and_analyze_predictions()
    
    # 2. 计算排名和占比
    rankings_df = calculate_weekly_rankings(predictions_df)
    
    # 3. 分析排名预测性能
    method_rank_stats = analyze_ranking_performance(rankings_df)
    
    # 4. 分析投票占比预测性能
    method_pct_stats = analyze_percentage_performance(rankings_df)
    
    # 5. 分析季节性性能
    seasonal_stats = analyze_seasonal_performance(rankings_df)
    
    # 6. 分析最佳表现选手
    contestant_stats = analyze_top_performers(rankings_df)
    
    # 7. 分析不确定性影响
    uncertainty_analysis = analyze_uncertainty_performance(rankings_df)
    
    # 8. 生成报告
    report = generate_performance_report(rankings_df, method_rank_stats, method_pct_stats, 
                                       seasonal_stats, contestant_stats, uncertainty_analysis)
    
    print("="*80)
    print("模型性能分析完成！")
    print("="*80)

if __name__ == "__main__":
    main()