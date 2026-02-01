import pandas as pd
import numpy as np

def fix_weight_values():
    """修复权重负值问题"""
    print("=== 修复权重负值问题 ===")
    
    # 加载数据
    df = pd.read_csv('ultimate_90_percent_scores.csv')
    
    # 创建原始权重列的副本（用于对比）
    df['judge_weight_original'] = df['judge_weight'].copy()
    df['fan_weight_original'] = df['fan_weight'].copy()
    
    # 只处理非NaN值
    mask = df['judge_weight'].notna() & df['fan_weight'].notna()
    
    print(f"总记录数: {len(df)}")
    print(f"有效权重记录数: {mask.sum()}")
    
    # 找出负值权重记录
    negative_judge = df[mask & (df['judge_weight'] < 0)]
    negative_fan = df[mask & (df['fan_weight'] < 0)]
    large_judge = df[mask & (df['judge_weight'] > 1)]
    
    print(f"评委权重负值数: {len(negative_judge)}")
    print(f"粉丝权重负值数: {len(negative_fan)}")
    print(f"评委权重大于1数: {len(large_judge)}")
    
    # 修复权重值
    # 方法1: 将负权重设为0，调整另一个权重
    def fix_pair(judge_weight, fan_weight):
        """修复一对权重值"""
        total = judge_weight + fan_weight
        
        if judge_weight < 0:
            judge_weight = 0
            fan_weight = total
        elif fan_weight < 0:
            fan_weight = 0
            judge_weight = total
        elif judge_weight > 1:
            judge_weight = 1
            fan_weight = total - 1
        elif fan_weight > 1:
            fan_weight = 1
            judge_weight = total - 1
            
        return judge_weight, fan_weight
    
    # 应用修复
    for idx in df[mask].index:
        judge = df.loc[idx, 'judge_weight']
        fan = df.loc[idx, 'fan_weight']
        fixed_judge, fixed_fan = fix_pair(judge, fan)
        df.loc[idx, 'judge_weight'] = fixed_judge
        df.loc[idx, 'fan_weight'] = fixed_fan
    
    # 验证修复
    fixed_mask = df['judge_weight'].notna() & df['fan_weight'].notna()
    fixed_negative_judge = df[fixed_mask & (df['judge_weight'] < 0)]
    fixed_negative_fan = df[fixed_mask & (df['fan_weight'] < 0)]
    fixed_large_judge = df[fixed_mask & (df['judge_weight'] > 1)]
    fixed_large_fan = df[fixed_mask & (df['fan_weight'] > 1)]
    
    print(f"\n=== 修复后验证 ===")
    print(f"评委权重负值数: {len(fixed_negative_judge)}")
    print(f"粉丝权重负值数: {len(fixed_negative_fan)}")
    print(f"评委权重大于1数: {len(fixed_large_judge)}")
    print(f"粉丝权重大于1数: {len(fixed_large_fan)}")
    
    # 计算修复前后的差异
    print(f"\n=== 修复前后对比 ===")
    df_fixed = df[mask]
    judge_diff = df_fixed['judge_weight'] - df_fixed['judge_weight_original']
    fan_diff = df_fixed['fan_weight'] - df_fixed['fan_weight_original']
    
    print(f"评委权重平均变化: {judge_diff.mean():.6f}")
    print(f"粉丝权重平均变化: {fan_diff.mean():.6f}")
    print(f"评委权重最大变化: {judge_diff.max():.6f}")
    print(f"粉丝权重最大变化: {fan_diff.max():.6f}")
    
    # 保存修复后的数据
    output_file = 'ultimate_90_percent_scores_fixed.csv'
    df.to_csv(output_file, index=False)
    print(f"\n修复后的数据已保存到: {output_file}")
    
    return df

if __name__ == "__main__":
    fixed_df = fix_weight_values()