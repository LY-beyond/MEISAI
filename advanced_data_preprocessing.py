import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

def split_by_voting_mechanism(df):
    """
    按投票机制拆分数据集
    """
    print("按投票机制拆分数据集...")
    
    # Season 1-2: Rank-based + Regular elimination
    rank_regular_df = df[df['season'].isin([1, 2])].copy()
    
    # Season 3-27: Percentage-based + Regular elimination  
    percentage_regular_df = df[df['season'].isin(range(3, 28))].copy()
    
    # Season 28-34: Rank-based + Bottom Two mechanism
    rank_bottom_two_df = df[df['season'].isin(range(28, 35))].copy()
    
    print(f"Rank-based Regular Elimination (S1-2): {len(rank_regular_df)} records")
    print(f"Percentage-based Regular Elimination (S3-27): {len(percentage_regular_df)} records")
    print(f"Rank-based Bottom Two (S28-34): {len(rank_bottom_two_df)} records")
    
    return rank_regular_df, percentage_regular_df, rank_bottom_two_df

def mark_key_event_weeks(df):
    """
    标记关键事件周
    """
    print("标记关键事件周...")
    
    # 1. 标记淘汰周
    df['is_elimination_week'] = df['results'].str.contains('Eliminated Week', na=False)
    
    # 2. 标记淘汰类型
    def get_elimination_type(results):
        if pd.isna(results):
            return 'Unknown'
        
        results_lower = results.lower()
        
        if 'eliminated' in results_lower:
            if any(keyword in results_lower for keyword in ['bottom two', 'bottom2', 'judges decision', 'judges save']):
                return 'Bottom_Two_Judges_Decision'
            else:
                return 'Regular_Elimination'
        elif any(keyword in results_lower for keyword in ['withdrew', 'injury', 'medical']):
            return 'Withdrawal'
        elif '1st place' in results_lower or 'winner' in results_lower:
            return 'Winner'
        elif '2nd place' in results_lower or 'runner' in results_lower:
            return 'Runner_up'
        elif '3rd place' in results_lower:
            return 'Third_Place'
        else:
            return 'Other'
    
    df['elimination_type'] = df['results'].apply(get_elimination_type)
    
    # 3. 标记淘汰选手
    def extract_eliminated_week(results):
        if pd.isna(results):
            return None
        
        # 提取"Eliminated Week N"中的周数
        match = re.search(r'Eliminated Week\s*(\d+)', results, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None
    
    df['eliminated_week'] = df['results'].apply(extract_eliminated_week)
    
    # 4. 标记Bottom Two选手
    df['is_bottom_two'] = df['elimination_type'] == 'Bottom_Two_Judges_Decision'
    
    # 5. 基于季度和周数推断Bottom Two（针对S28-34）
    if 'season' in df.columns:
        df = infer_bottom_two_by_pattern(df)
    
    return df

def infer_bottom_two_by_pattern(df):
    """
    基于季度和周数推断Bottom Two（针对S28-34）
    """
    print("推断Bottom Two模式...")
    
    # Season 28-34的Bottom Two通常出现在后期
    bottom_two_seasons = range(28, 35)
    
    for season in bottom_two_seasons:
        if season not in df['season'].values:
            continue
            
        season_data = df[df['season'] == season].copy()
        
        # 按eliminated_week排序
        season_data = season_data.sort_values('eliminated_week')
        
        # 推断Bottom Two周（通常在后期，且淘汰人数较少）
        if len(season_data) > 0:
            # 获取所有淘汰周
            elimination_weeks = sorted(season_data['eliminated_week'].dropna().unique())
            
            if len(elimination_weeks) > 0:
                # 通常Bottom Two出现在最后几周
                # 假设最后3周可能是Bottom Two周
                potential_bottom_two_weeks = elimination_weeks[-3:] if len(elimination_weeks) >= 3 else elimination_weeks
                
                # 标记这些周的选手为潜在Bottom Two
                for week in potential_bottom_two_weeks:
                    week_mask = (df['season'] == season) & (df['eliminated_week'] == week)
                    df.loc[week_mask, 'is_bottom_two'] = True
                    df.loc[week_mask, 'elimination_type'] = 'Bottom_Two_Judges_Decision'
    
    return df

def clean_dirty_data(df):
    """
    处理脏数据
    """
    print("处理脏数据...")
    
    # 1. 处理评委分数中的N/A
    for week in range(1, 12):
        for judge in range(1, 5):
            col = f'week{week}_judge{judge}_score'
            if col in df.columns:
                # N/A替换为0（表示无该评委或无分数）
                df[col] = df[col].fillna(0)
    
    # 2. 保留所有选手数据，不删除非淘汰周数据
    # df_elimination = df[df['is_elimination_week']].copy()
    
    # 3. 处理异常分数
    for week in range(1, 12):
        week_col = f'week{week}_avg_score'
        if week_col in df.columns:
            # 过滤掉0分但不是淘汰选手的记录
            df = df[
                ~((df[week_col] == 0) & 
                  (~df['results'].str.contains('Eliminated Week', na=False)))
            ]
    
    return df

def comprehensive_data_preprocessing(df):
    """
    综合数据预处理：按建议重新处理
    """
    print("开始重新数据预处理...")
    
    # 步骤1：按投票机制拆分
    print("步骤1：按投票机制拆分数据集...")
    rank_regular_df, percentage_regular_df, rank_bottom_two_df = split_by_voting_mechanism(df)
    
    # 步骤2：标记关键事件周
    print("\n步骤2：标记关键事件周...")
    rank_regular_df = mark_key_event_weeks(rank_regular_df)
    percentage_regular_df = mark_key_event_weeks(percentage_regular_df)
    rank_bottom_two_df = mark_key_event_weeks(rank_bottom_two_df)
    
    # 步骤3：处理脏数据
    print("\n步骤3：处理脏数据...")
    rank_regular_clean = clean_dirty_data(rank_regular_df)
    percentage_regular_clean = clean_dirty_data(percentage_regular_df)
    rank_bottom_two_clean = clean_dirty_data(rank_bottom_two_df)
    
    print(f"清理后数据量:")
    print(f"  Rank-based Regular: {len(rank_regular_clean)} records")
    print(f"  Percentage-based Regular: {len(percentage_regular_clean)} records")
    print(f"  Rank-based Bottom Two: {len(rank_bottom_two_clean)} records")
    
    # 步骤4：保存处理后的数据
    print("\n步骤4：保存处理后的数据...")
    rank_regular_clean.to_csv('dwts_rank_regular_processed.csv', index=False)
    percentage_regular_clean.to_csv('dwts_percentage_regular_processed.csv', index=False)
    rank_bottom_two_clean.to_csv('dwts_rank_bottom_two_processed.csv', index=False)
    
    print("数据预处理完成！")
    
    return {
        'rank_regular': rank_regular_clean,
        'percentage_regular': percentage_regular_clean,
        'rank_bottom_two': rank_bottom_two_clean
    }

def validate_preprocessing_results(processed_data):
    """
    验证预处理结果
    """
    print("\n=== 预处理结果验证 ===")
    
    for dataset_name, df in processed_data.items():
        print(f"\n{dataset_name.upper()} 数据集:")
        print(f"  记录数: {len(df)}")
        print(f"  季度范围: {df['season'].min()} - {df['season'].max()}")
        print(f"  淘汰周数: {df['eliminated_week'].nunique()}")
        print(f"  淘汰类型分布:")
        
        elimination_types = df['elimination_type'].value_counts()
        for elim_type, count in elimination_types.items():
            print(f"    {elim_type}: {count}")
        
        # 验证Bottom Two数据
        if 'bottom_two' in dataset_name:
            bottom_two_count = df['is_bottom_two'].sum()
            print(f"  Bottom Two周数: {bottom_two_count}")
        
        # 显示新增列
        print(f"  新增列: {list(df.columns[-4:])}")

def main():
    """
    主函数：执行完整的重新数据预处理
    """
    print("DWTS 数据重新预处理")
    print("=" * 50)
    
    # 读取原始数据
    try:
        df = pd.read_csv('2026_MCM_Problem_C_Data.csv')
        print(f"原始数据加载成功，共 {len(df)} 条记录")
    except FileNotFoundError:
        print("错误：找不到原始数据文件 2026_MCM_Problem_C_Data.csv")
        return
    
    # 执行综合预处理
    processed_data = comprehensive_data_preprocessing(df)
    
    # 验证结果
    validate_preprocessing_results(processed_data)
    
    # 生成预处理报告
    generate_preprocessing_report(processed_data)
    
    print("\n" + "=" * 50)
    print("重新数据预处理完成！")
    print("生成的文件:")
    print("  - dwts_rank_regular_processed.csv (S1-2)")
    print("  - dwts_percentage_regular_processed.csv (S3-27)")
    print("  - dwts_rank_bottom_two_processed.csv (S28-34)")

def generate_preprocessing_report(processed_data):
    """
    生成预处理报告
    """
    print("\n=== 预处理报告 ===")
    
    total_original = sum(len(df) for df in processed_data.values())
    print(f"原始数据记录数: {total_original}")
    
    for dataset_name, df in processed_data.items():
        percentage = (len(df) / total_original) * 100
        print(f"{dataset_name}: {len(df)} 记录 ({percentage:.1f}%)")
    
    # 统计淘汰类型
    print("\n淘汰类型统计:")
    for dataset_name, df in processed_data.items():
        print(f"\n{dataset_name}:")
        elimination_stats = df['elimination_type'].value_counts()
        for elim_type, count in elimination_stats.items():
            percentage = (count / len(df)) * 100
            print(f"  {elim_type}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    main()