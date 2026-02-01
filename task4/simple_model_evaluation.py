#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单模型性能评估脚本
"""

import pandas as pd
import numpy as np
import joblib

def main():
    print("="*80)
    print("90%+准确率终极模型性能评估")
    print("="*80)
    
    # 1. 加载数据
    print("加载数据...")
    ultimate_scores = pd.read_csv('task4/ultimate_90_percent_scores.csv')
    fan_df = pd.read_csv('task1/fan_vote_predictions_enhanced.csv')
    
    print(f"终极模型评分数据: {len(ultimate_scores)}条记录")
    print(f"粉丝投票数据: {len(fan_df)}条记录")
    
    # 2. 计算实际淘汰
    print("\n计算实际淘汰情况...")
    actual_eliminations = {}
    
    for (season, week), week_group in fan_df.groupby(['season', 'week']):
        if week < fan_df['week'].max():
            next_week_group = fan_df[(fan_df['season'] == season) & (fan_df['week'] == week + 1)]
            current_contestants = set(week_group['contestant'])
            next_contestants = set(next_week_group['contestant'])
            
            eliminated = list(current_contestants - next_contestants)
            if eliminated:
                actual_eliminations[(season, week)] = eliminated
    
    print(f"实际淘汰周数: {len(actual_eliminations)}")
    
    # 3. 计算模型预测
    print("\n计算模型预测结果...")
    predicted_eliminations = {}
    
    for (season, week), week_group in ultimate_scores.groupby(['season', 'week']):
        if week < ultimate_scores['week'].max():
            # 根据淘汰概率预测淘汰选手
            week_group = week_group.copy()
            
            # 计算综合得分
            week_group['normalized_rank'] = 1 - (week_group['rank'] - 1) / (len(week_group) - 1)
            week_group['elimination_score'] = (
                week_group['elimination_probability'] * 0.7 + 
                week_group['normalized_rank'] * 0.3
            )
            
            # 预测淘汰选手（得分最高的1名）
            num_to_eliminate = 1
            top_elimination = week_group.nlargest(num_to_eliminate, 'elimination_score')
            predicted = list(top_elimination['celebrity_name'])
            
            predicted_eliminations[(season, week)] = predicted
    
    print(f"模型预测周数: {len(predicted_eliminations)}")
    
    # 4. 评估准确率
    print("\n评估准确率...")
    matches = 0
    total_weeks = len(actual_eliminations)
    
    for key in actual_eliminations:
        if key in predicted_eliminations:
            actual_set = set(actual_eliminations[key])
            pred_set = set(predicted_eliminations[key])
            
            # 检查预测是否正确
            if pred_set.issubset(actual_set) or actual_set == pred_set:
                matches += 1
    
    accuracy = matches / total_weeks if total_weeks > 0 else 0
    
    print(f"总周数: {total_weeks}")
    print(f"正确预测周数: {matches}")
    print(f"每周准确率: {accuracy*100:.2f}%")
    
    # 5. 按赛季统计
    print("\n按赛季准确率统计:")
    season_results = []
    
    for (season, week) in actual_eliminations:
        if (season, week) in predicted_eliminations:
            actual_set = set(actual_eliminations[(season, week)])
            pred_set = set(predicted_eliminations[(season, week)])
            is_correct = pred_set.issubset(actual_set) or actual_set == pred_set
            season_results.append({'season': season, 'is_correct': is_correct})
    
    season_df = pd.DataFrame(season_results)
    season_stats = season_df.groupby('season').agg({
        'is_correct': ['count', 'sum', 'mean']
    }).round(3)
    
    season_stats.columns = ['total_weeks', 'correct_weeks', 'accuracy']
    print(season_stats)
    
    # 6. 错误案例分析
    print("\n错误案例分析:")
    error_cases = []
    
    for key in actual_eliminations:
        if key in predicted_eliminations:
            actual_set = set(actual_eliminations[key])
            pred_set = set(predicted_eliminations[key])
            
            if not (pred_set.issubset(actual_set) or actual_set == pred_set):
                error_cases.append({
                    'season': key[0],
                    'week': key[1],
                    'actual': list(actual_set),
                    'predicted': list(pred_set)
                })
    
    print(f"错误案例数: {len(error_cases)}")
    
    if len(error_cases) > 0:
        print("\n前5个错误案例:")
        for i, case in enumerate(error_cases[:5], 1):
            print(f"  {i}. 赛季{case['season']}第{case['week']}周:")
            print(f"     实际淘汰: {case['actual']}")
            print(f"     预测淘汰: {case['predicted']}")
    
    # 7. 模型总结
    print("\n" + "="*80)
    print("模型性能总结:")
    print("="*80)
    print(f"最终准确率: {accuracy*100:.2f}%")
    print(f"目标准确率: 90%+")
    print(f"达成情况: {'✅ 达成目标' if accuracy >= 0.9 else '❌ 未达成目标'}")
    
    if accuracy >= 0.9:
        print(f"超额完成: {accuracy*100 - 90:.2f}个百分点")
    
    # 8. 与历史模型对比
    print("\n与历史模型对比:")
    print("原始模型准确率: 25.52%")
    print("优化模型准确率: 29.72%")
    print("改进模型准确率: 70.63%")
    print("增强模型准确率: 72.38%")
    print(f"终极模型准确率: {accuracy*100:.2f}%")
    
    improvement = (accuracy*100 - 25.52) / 25.52 * 100
    print(f"总提升幅度: {improvement:.1f}%")
    
    print("\n" + "="*80)
    print("评估完成!")
    print("="*80)
    
    return accuracy

if __name__ == "__main__":
    accuracy = main()
    
    # 保存评估结果
    with open('task4/model_evaluation_summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"模型评估结果\n")
        f.write(f"============\n")
        f.write(f"评估时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"最终准确率: {accuracy*100:.2f}%\n")
        f.write(f"目标达成: {'是' if accuracy >= 0.9 else '否'}\n")
        f.write(f"总提升幅度: {(accuracy*100 - 25.52) / 25.52 * 100:.1f}%\n")
    
    print(f"\n评估结果已保存到 task4/model_evaluation_summary.txt")