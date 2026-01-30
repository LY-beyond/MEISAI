import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_and_validate_data(file_path='2026_MCM_Problem_C_Data.csv'):
    """
    加载并验证DWTS数据
    """
    print("正在加载数据...")
    df = pd.read_csv(file_path)
    
    print(f"数据形状: {df.shape}")
    print(f"列数: {len(df.columns)}")
    print(f"行数: {len(df)}")
    
    # 基本信息检查
    print("\n数据基本信息:")
    print(df.info())
    
    # 缺失值统计
    print("\n缺失值统计:")
    missing_stats = df.isnull().sum()
    missing_percentage = (missing_stats / len(df)) * 100
    missing_df = pd.DataFrame({
        '缺失数量': missing_stats,
        '缺失百分比': missing_percentage
    })
    print(missing_df[missing_df['缺失数量'] > 0].head(10))
    
    return df

def clean_basic_data(df):
    """
    基础数据清洗
    """
    print("\n=== 基础数据清洗 ===")
    
    # 1. 处理缺失值
    # 对于数值型列，将N/A替换为0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # 对于文本列，将N/A替换为空字符串
    text_cols = df.select_dtypes(include=['object']).columns
    df[text_cols] = df[text_cols].fillna('')
    
    # 2. 数据类型转换
    # 确保年龄为数值类型
    if 'celebrity_age_during_season' in df.columns:
        df['celebrity_age_during_season'] = pd.to_numeric(
            df['celebrity_age_during_season'], errors='coerce'
        ).fillna(0)
    
    # 3. 清理文本数据
    # 去除首尾空格
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    print("基础数据清洗完成")
    return df

def process_judge_scores(df):
    """
    处理评委分数
    """
    print("\n=== 评委分数处理 ===")
    
    # 1. 计算每周总分
    for week in range(1, 12):
        judge_cols = [f'week{week}_judge{i}_score' for i in range(1, 5)]
        
        # 检查这些列是否存在
        existing_cols = [col for col in judge_cols if col in df.columns]
        
        if existing_cols:
            # 计算每周总分（只考虑有效分数）
            def calculate_weekly_total(row):
                scores = row[existing_cols]
                # 过滤掉0分和负分（这些通常是淘汰后的标记）
                valid_scores = scores[(scores > 0) & (scores <= 40)]  # 最高分40（4个评委各10分）
                return valid_scores.sum() if len(valid_scores) > 0 else 0
            
            df[f'week{week}_total_score'] = df.apply(calculate_weekly_total, axis=1)
            
            # 计算每周平均分
            def calculate_weekly_average(row):
                scores = row[existing_cols]
                valid_scores = scores[(scores > 0) & (scores <= 40)]
                return valid_scores.mean() if len(valid_scores) > 0 else 0
            
            df[f'week{week}_avg_score'] = df.apply(calculate_weekly_average, axis=1)
    
    # 2. 计算选手整体表现
    week_total_cols = [f'week{i}_total_score' for i in range(1, 12) if f'week{i}_total_score' in df.columns]
    week_avg_cols = [f'week{i}_avg_score' for i in range(1, 12) if f'week{i}_avg_score' in df.columns]
    
    # 计算总平均分（排除0分）
    def calculate_overall_average(row):
        scores = row[week_avg_cols]
        valid_scores = scores[scores > 0]
        return valid_scores.mean() if len(valid_scores) > 0 else 0
    
    df['overall_avg_score'] = df.apply(calculate_overall_average, axis=1)
    
    # 计算总分（所有有效周的分数之和）
    def calculate_total_points(row):
        scores = row[week_total_cols]
        valid_scores = scores[scores > 0]
        return valid_scores.sum() if len(valid_scores) > 0 else 0
    
    df['total_points'] = df.apply(calculate_total_points, axis=1)
    
    # 3. 计算参赛周数
    def calculate_weeks_participated(row):
        scores = row[week_total_cols]
        return (scores > 0).sum()
    
    df['weeks_participated'] = df.apply(calculate_weeks_participated, axis=1)
    
    print("评委分数处理完成")
    return df

def create_features(df):
    """
    创建衍生特征
    """
    print("\n=== 特征工程 ===")
    
    # 1. 选手年龄组
    def categorize_age(age):
        if age == 0:
            return 'Unknown'
        elif age < 25:
            return '<25'
        elif age < 35:
            return '25-34'
        elif age < 45:
            return '35-44'
        elif age < 55:
            return '45-54'
        else:
            return '55+'
    
    if 'celebrity_age_during_season' in df.columns:
        df['age_group'] = df['celebrity_age_during_season'].apply(categorize_age)
    
    # 2. 选手行业标准化
    def standardize_industry(industry):
        if pd.isna(industry) or industry == '':
            return 'Unknown'
        
        industry_lower = industry.lower().strip()
        
        # 标准化常见行业
        industry_mapping = {
            'athlete': ['athlete', 'athletes', 'sports', 'football', 'basketball', 'baseball', 'hockey', 'tennis', 'gymnast'],
            'model': ['model', 'models', 'fashion'],
            'actor': ['actor', 'actress', 'actors', 'actresses', 'entertainment', 'television', 'film'],
            'musician': ['musician', 'singer', 'singers', 'music', 'rapper', 'rapper/actor'],
            'tv_personality': ['tv personality', 'television personality', 'reality tv', 'reality star'],
            'politician': ['politician', 'politicians', 'political', 'congresswoman', 'senator'],
            'news': ['news anchor', 'news', 'journalist', 'reporter'],
            'dancer': ['dancer', 'professional dancer', 'ballroom dancer'],
            'comedian': ['comedian', 'comedy'],
            'business': ['businessman', 'businesswoman', 'entrepreneur', 'business'],
            'royalty': ['royalty', 'princess', 'prince', 'duke', 'duchess'],
            'other': ['other', 'miscellaneous', 'various']
        }
        
        for standard, variants in industry_mapping.items():
            for variant in variants:
                if variant in industry_lower:
                    return standard.title()
        
        return industry.title()  # 保持原样但首字母大写
    
    if 'celebrity_industry' in df.columns:
        df['standardized_industry'] = df['celebrity_industry'].apply(standardize_industry)
    
    # 3. 国籍标准化
    def standardize_country(country):
        if pd.isna(country) or country == '':
            return 'Unknown'
        
        country_lower = country.lower().strip()
        
        # 标准化常见国家
        country_mapping = {
            'USA': ['united states', 'us', 'usa', 'american'],
            'UK': ['united kingdom', 'uk', 'british', 'england', 'britain'],
            'Australia': ['australia', 'australian'],
            'Canada': ['canada', 'canadian'],
            'Germany': ['germany', 'german'],
            'France': ['france', 'french'],
            'Italy': ['italy', 'italian'],
            'Spain': ['spain', 'spanish'],
            'Brazil': ['brazil', 'brazilian'],
            'Other': ['other', 'unknown']
        }
        
        for standard, variants in country_mapping.items():
            for variant in variants:
                if variant in country_lower:
                    return standard
        
        return country.title()
    
    if 'celebrity_homecountry/region' in df.columns:
        df['standardized_country'] = df['celebrity_homecountry/region'].apply(standardize_country)
    
    # 4. 淘汰状态
    def get_elimination_status(results):
        if pd.isna(results) or results == '':
            return 'Unknown'
        
        results_lower = results.lower()
        
        if 'eliminated' in results_lower:
            # 提取淘汰周数
            for week in range(1, 12):
                if f'week {week}' in results_lower or f'week{week}' in results_lower:
                    return f'Eliminated Week {week}'
            return 'Eliminated'
        elif '1st place' in results_lower or 'winner' in results_lower:
            return 'Winner'
        elif '2nd place' in results_lower or 'runner' in results_lower:
            return 'Runner-up'
        elif '3rd place' in results_lower:
            return '3rd Place'
        elif 'withdrew' in results_lower or 'injury' in results_lower:
            return 'Withdrew'
        else:
            return 'Other'
    
    if 'results' in df.columns:
        df['elimination_status'] = df['results'].apply(get_elimination_status)
    
    # 5. 表现趋势（最后几周的平均分 vs 整体平均分）
    def calculate_performance_trend(row):
        recent_weeks = ['week8_avg_score', 'week9_avg_score', 'week10_avg_score', 'week11_avg_score']
        recent_weeks = [col for col in recent_weeks if col in df.columns]
        
        if not recent_weeks:
            return 0
        
        recent_scores = row[recent_weeks]
        valid_recent = recent_scores[recent_scores > 0]
        
        if len(valid_recent) == 0:
            return 0
        
        recent_avg = valid_recent.mean()
        overall_avg = row['overall_avg_score'] if 'overall_avg_score' in df.columns else 0
        
        if overall_avg == 0:
            return 0
        
        return (recent_avg - overall_avg) / overall_avg * 100
    
    if 'overall_avg_score' in df.columns:
        df['performance_trend'] = df.apply(calculate_performance_trend, axis=1)
    
    print("特征工程完成")
    return df

def normalize_data(df):
    """
    数据标准化
    """
    print("\n=== 数据标准化 ===")
    
    # 1. 标准化分数到0-100范围
    if 'overall_avg_score' in df.columns:
        max_score = df['overall_avg_score'].max()
        if max_score > 0:
            df['normalized_score'] = (df['overall_avg_score'] / max_score) * 100
        else:
            df['normalized_score'] = 0
    
    # 2. 季度标准化（处理不同季度的评分差异）
    if 'season' in df.columns and 'overall_avg_score' in df.columns:
        season_averages = df.groupby('season')['overall_avg_score'].mean()
        
        def season_normalize(row):
            season_avg = season_averages.get(row['season'], 0)
            if season_avg == 0:
                return 0
            return (row['overall_avg_score'] / season_avg) * 50  # 以50为基准
        
        df['season_normalized_score'] = df.apply(season_normalize, axis=1)
    
    # 3. 创建表现等级
    def categorize_performance(score):
        if score == 0:
            return 'Not Participated'
        elif score < 20:
            return 'Poor'
        elif score < 40:
            return 'Below Average'
        elif score < 60:
            return 'Average'
        elif score < 80:
            return 'Good'
        elif score < 95:
            return 'Excellent'
        else:
            return 'Outstanding'
    
    if 'normalized_score' in df.columns:
        df['performance_level'] = df['normalized_score'].apply(categorize_performance)
    
    print("数据标准化完成")
    return df

def validate_and_save(df, output_file='dwts_processed_data.csv'):
    """
    数据验证和保存
    """
    print("\n=== 数据验证 ===")
    
    # 1. 基本统计信息
    print("处理后数据统计:")
    print(f"总行数: {len(df)}")
    print(f"总列数: {len(df.columns)}")
    
    # 2. 检查关键字段
    key_fields = ['celebrity_name', 'ballroom_partner', 'season', 'overall_avg_score']
    for field in key_fields:
        if field in df.columns:
            valid_count = df[field].notna().sum()
            print(f"{field}: {valid_count}/{len(df)} ({valid_count/len(df)*100:.1f}%)")
    
    # 3. 检查分数范围
    if 'overall_avg_score' in df.columns:
        valid_scores = df[df['overall_avg_score'] > 0]['overall_avg_score']
        print(f"有效平均分范围: {valid_scores.min():.2f} - {valid_scores.max():.2f}")
        print(f"平均分中位数: {valid_scores.median():.2f}")
    
    # 4. 保存处理后的数据
    print(f"\n保存处理后的数据到 {output_file}")
    df.to_csv(output_file, index=False)
    print("数据保存完成!")
    
    return df

def main():
    """
    主函数：执行完整的数据预处理流程
    """
    print("开始DWTS数据预处理...")
    print("=" * 50)
    
    # 1. 加载和验证数据
    df = load_and_validate_data()
    
    # 2. 基础数据清洗
    df = clean_basic_data(df)
    
    # 3. 处理评委分数
    df = process_judge_scores(df)
    
    # 4. 特征工程
    df = create_features(df)
    
    # 5. 数据标准化
    df = normalize_data(df)
    
    # 6. 数据验证和保存
    df = validate_and_save(df)
    
    print("\n" + "=" * 50)
    print("数据预处理完成!")
    print(f"最终数据形状: {df.shape}")
    
    return df

if __name__ == "__main__":
    processed_df = main()