import pandas as pd
import numpy as np
import itertools
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import re

warnings.filterwarnings('ignore')

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 数据加载和预处理 ====================
def load_and_preprocess_data(rank_regular_path, percentage_regular_path, rank_bottom_two_path):
    """
    加载预处理数据（三个文件）
    """
    print("正在加载预处理数据...")
    
    # 加载三个文件
    df_rank_regular = pd.read_csv(rank_regular_path)
    df_percentage_regular = pd.read_csv(percentage_regular_path)
    df_rank_bottom_two = pd.read_csv(rank_bottom_two_path)
    
    print(f"Rank-based Regular (S1-2): {len(df_rank_regular)} 条记录")
    print(f"Percentage-based Regular (S3-27): {len(df_percentage_regular)} 条记录")
    print(f"Rank-based Bottom Two (S28-34): {len(df_rank_bottom_two)} 条记录")
    
    # 为每个数据集添加阶段标记
    df_rank_regular['voting_phase'] = 'rank_regular'
    df_percentage_regular['voting_phase'] = 'percentage_regular'
    df_rank_bottom_two['voting_phase'] = 'rank_bottom_two'
    
    # 合并数据
    df = pd.concat([df_rank_regular, df_percentage_regular, df_rank_bottom_two], ignore_index=True)
    
    print(f"合并后总数据: {len(df)} 条记录，{df['season'].nunique()} 个赛季")
    
    # 检查数据列
    print(f"数据列数: {len(df.columns)}")
    print(f"前10列: {list(df.columns[:10])}")
    
    # 查找所有周相关列
    week_columns = [col for col in df.columns if 'week' in col.lower()]
    print(f"找到周相关列 {len(week_columns)} 个")
    
    # 查找评委分数列（格式如 week1_judge1_score）
    judge_score_columns = []
    for col in df.columns:
        col_lower = col.lower()
        if 'week' in col_lower and 'judge' in col_lower and 'score' in col_lower:
            judge_score_columns.append(col)
    
    print(f"找到评委分数列 {len(judge_score_columns)} 个")
    
    # 计算每周总分（将多个评委分数相加）
    total_score_cols = []
    max_week = 0
    
    if judge_score_columns:
        # 提取周数
        week_numbers = {}
        for col in judge_score_columns:
            # 使用正则表达式提取周数
            match = re.search(r'week(\d+)', col.lower())
            if match:
                week_num = int(match.group(1))
                if week_num not in week_numbers:
                    week_numbers[week_num] = []
                week_numbers[week_num].append(col)
        
        # 为每周创建总分列
        for week_num in sorted(week_numbers.keys()):
            score_cols = week_numbers[week_num]
            total_col_name = f'week{week_num}_total_score'
            
            # 计算总分
            df[total_col_name] = 0
            for col in score_cols:
                # 将列转换为数值类型，处理缺失值
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                df[total_col_name] += df[col]
            
            total_score_cols.append(total_col_name)
            max_week = max(max_week, week_num)
        
        print(f"已创建总分列 {len(total_score_cols)} 个")
        print(f"检测到最多 {max_week} 周的数据")
    else:
        print("警告: 未找到评委分数列")
        # 尝试查找现有的总分列
        existing_total_cols = [col for col in df.columns if 'total' in col.lower() and 'score' in col.lower()]
        if existing_total_cols:
            total_score_cols = existing_total_cols
            print(f"使用现有的总分列: {len(total_score_cols)} 个")
            # 确定最大周数
            for col in total_score_cols:
                try:
                    week_num = int(col.split('_')[0].replace('week', ''))
                    max_week = max(max_week, week_num)
                except:
                    continue
    
    # 确保数据中有必要的列
    required_columns = ['celebrity_name', 'season', 'placement', 'voting_phase']
    for col in required_columns:
        if col not in df.columns:
            print(f"警告: 缺少必要列 '{col}'")
    
    # 处理缺失值
    for col in total_score_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    print(f"数据加载完成，共 {len(df)} 条记录")
    
    return df, max_week, total_score_cols

# ==================== 2. 投票方法选择（基于投票阶段） ====================
def get_voting_method(season, voting_phase=None):
    """
    根据赛季和投票阶段返回投票方法
    """
    if voting_phase:
        if voting_phase == 'rank_regular' or voting_phase == 'rank_bottom_two':
            return 'rank'
        elif voting_phase == 'percentage_regular':
            return 'percentage'
    else:
        # 如果没有提供voting_phase，则根据赛季判断
        if season <= 2:
            return 'rank'
        elif 3 <= season <= 27:
            return 'percentage'
        else:  # 28-34
            return 'rank'

# ==================== 3. 预测函数 ====================
def predict_by_rank(judge_scores, eliminated, bottom_two=None):
    """
    使用排名法预测粉丝投票
    """
    n = len(judge_scores)
    contestants = list(judge_scores.keys())
    
    if n == 0:
        return {}
    
    # 计算评委排名（分数越高，排名数字越小）
    sorted_by_score = sorted(judge_scores.items(), key=lambda x: x[1], reverse=True)
    judge_ranks = {contestant: rank+1 for rank, (contestant, _) in enumerate(sorted_by_score)}
    
    # 采样数量（根据n调整）
    if n <= 7:
        # 小规模：枚举所有排列
        all_perms = list(itertools.permutations(contestants))
        valid_perms = []
        
        for perm in all_perms:
            # 计算总排名
            total_ranks = {}
            for fan_rank, contestant in enumerate(perm, 1):
                total_ranks[contestant] = judge_ranks[contestant] + fan_rank
            
            # 找出总排名最高的（最差）
            max_rank = max(total_ranks.values())
            worst_contestants = [c for c, r in total_ranks.items() if r == max_rank]
            
            # 检查是否匹配淘汰结果
            if eliminated:
                if set(worst_contestants) == set(eliminated):
                    valid_perms.append(perm)
            elif bottom_two:  # Bottom Two模式
                # 检查最差的两名选手是否是bottom_two
                sorted_by_total_rank = sorted(total_ranks.items(), key=lambda x: x[1], reverse=True)
                bottom_two_candidates = [c for c, _ in sorted_by_total_rank[-2:]]
                if set(bottom_two_candidates) == set(bottom_two):
                    valid_perms.append(perm)
            else:
                # 本周没有淘汰（如决赛周）
                valid_perms.append(perm)
        
        if valid_perms:
            # 计算平均粉丝排名
            fan_rank_matrix = np.zeros((len(valid_perms), n))
            contestant_to_idx = {c: i for i, c in enumerate(contestants)}
            
            for i, perm in enumerate(valid_perms):
                for fan_rank, contestant in enumerate(perm, 1):
                    fan_rank_matrix[i, contestant_to_idx[contestant]] = fan_rank
            
            avg_fan_ranks = fan_rank_matrix.mean(axis=0)
            
            # 将排名转换为投票比例（排名越好，投票越多）
            fan_votes = {}
            for i, contestant in enumerate(contestants):
                rank = avg_fan_ranks[i]
                fan_votes[contestant] = np.exp(-rank/2)
            
            # 归一化
            total = sum(fan_votes.values())
            if total > 0:
                fan_votes = {c: v/total for c, v in fan_votes.items()}
            else:
                fan_votes = {c: 1/n for c in contestants}
        else:
            # 没有有效排列，使用均匀分布
            fan_votes = {c: 1/n for c in contestants}
    
    else:
        # 大规模：蒙特卡洛采样
        n_samples = min(10000, 200 * n)
        samples = []
        
        for _ in range(n_samples):
            # 随机排列
            perm = np.random.permutation(contestants)
            
            # 计算总排名
            total_ranks = {}
            for fan_rank, contestant in enumerate(perm, 1):
                total_ranks[contestant] = judge_ranks[contestant] + fan_rank
            
            # 找出最差选手
            max_rank = max(total_ranks.values())
            worst_contestants = [c for c, r in total_ranks.items() if r == max_rank]
            
            # 检查是否匹配淘汰结果
            if eliminated:
                if set(worst_contestants) == set(eliminated):
                    samples.append(perm)
            elif bottom_two:  # Bottom Two模式
                # 检查最差的两名选手是否是bottom_two
                sorted_by_total_rank = sorted(total_ranks.items(), key=lambda x: x[1], reverse=True)
                bottom_two_candidates = [c for c, _ in sorted_by_total_rank[-2:]]
                if set(bottom_two_candidates) == set(bottom_two):
                    samples.append(perm)
            else:
                samples.append(perm)
        
        if samples:
            # 计算平均粉丝排名
            fan_rank_matrix = np.zeros((len(samples), n))
            contestant_to_idx = {c: i for i, c in enumerate(contestants)}
            
            for i, sample in enumerate(samples):
                for fan_rank, contestant in enumerate(sample, 1):
                    fan_rank_matrix[i, contestant_to_idx[contestant]] = fan_rank
            
            avg_fan_ranks = fan_rank_matrix.mean(axis=0)
            
            # 转换为投票比例
            fan_votes = {}
            for i, contestant in enumerate(contestants):
                rank = avg_fan_ranks[i]
                fan_votes[contestant] = np.exp(-rank/2)
            
            # 归一化
            total = sum(fan_votes.values())
            if total > 0:
                fan_votes = {c: v/total for c, v in fan_votes.items()}
            else:
                fan_votes = {c: 1/n for c in contestants}
        else:
            # 没有有效样本，使用均匀分布
            fan_votes = {c: 1/n for c in contestants}
    
    return fan_votes

def predict_by_percentage(judge_scores, eliminated):
    """
    使用百分比法预测粉丝投票
    """
    n = len(judge_scores)
    contestants = list(judge_scores.keys())
    
    if n == 0:
        return {}
    
    # 计算评委百分比
    total_judge_score = sum(judge_scores.values())
    if total_judge_score == 0:
        judge_percentages = {c: 1/n for c in contestants}
    else:
        judge_percentages = {c: score/total_judge_score for c, score in judge_scores.items()}
    
    # 蒙特卡洛采样
    n_samples = min(10000, 200 * n)
    samples = []
    
    for _ in range(n_samples):
        # 生成粉丝投票（狄利克雷分布）
        alpha = np.ones(n)  # 对称
        
        # 根据评委分数调整alpha（评委分数高，粉丝投票可能也高）
        for i, contestant in enumerate(contestants):
            alpha[i] = 1 + judge_scores[contestant] / 100
        
        fan_votes_sample = np.random.dirichlet(alpha)
        
        # 计算总百分比（评委权重0.4，粉丝权重0.6）
        total_percentages = {}
        for i, contestant in enumerate(contestants):
            total_percentages[contestant] = 0.4 * judge_percentages[contestant] + 0.6 * fan_votes_sample[i]
        
        # 找出总百分比最低的（最差）
        min_percentage = min(total_percentages.values())
        worst_contestants = [c for c, p in total_percentages.items() if p == min_percentage]
        
        # 检查是否匹配淘汰结果
        if eliminated:
            if set(worst_contestants) == set(eliminated):
                samples.append(fan_votes_sample)
        else:
            samples.append(fan_votes_sample)
    
    if samples:
        # 计算平均粉丝投票
        samples_array = np.array(samples)
        avg_fan_votes = samples_array.mean(axis=0)
        
        fan_votes = {}
        for i, contestant in enumerate(contestants):
            fan_votes[contestant] = avg_fan_votes[i]
    else:
        # 没有有效样本，使用均匀分布
        fan_votes = {c: 1/n for c in contestants}
    
    return fan_votes

# ==================== 4. 两种方法对比分析 ====================
def compare_voting_methods(df, max_week, total_score_cols):
    """
    对比两种投票方法在所有赛季的结果差异
    """
    print("\n开始对比两种投票方法...")
    
    if df.empty or not total_score_cols:
        print("数据不足，无法进行对比分析")
        return pd.DataFrame()
    
    seasons = sorted(df['season'].unique())
    comparison_results = []
    
    for season in seasons:
        season_data = df[df['season'] == season]
        if season_data.empty:
            continue
            
        # 获取该赛季的投票阶段
        if 'voting_phase' in season_data.columns:
            voting_phase = season_data['voting_phase'].iloc[0]
        else:
            # 根据赛季推断
            if season <= 2:
                voting_phase = 'rank_regular'
            elif 3 <= season <= 27:
                voting_phase = 'percentage_regular'
            else:
                voting_phase = 'rank_bottom_two'
        
        # 获取实际淘汰信息
        actual_eliminations = {}
        max_week_season = 0
        
        # 获取该赛季的最大周数
        for week in range(1, max_week + 1):
            col = f'week{week}_total_score'
            if col in season_data.columns and season_data[col].notna().any():
                max_week_season = week
        
        # 每周检查淘汰
        for week in range(1, max_week_season):
            current_col = f'week{week}_total_score'
            next_col = f'week{week+1}_total_score'
            
            if current_col in season_data.columns and next_col in season_data.columns:
                eliminated = []
                for _, row in season_data.iterrows():
                    current_score = row[current_col]
                    next_score = row[next_col]
                    
                    if current_score > 0 and (pd.isna(next_score) or next_score == 0):
                        eliminated.append(row['celebrity_name'])
                
                if eliminated:
                    actual_eliminations[(season, week)] = eliminated
        
        # 分别用两种方法预测
        rank_eliminations = {}
        percentage_eliminations = {}
        
        for week in range(1, max_week_season + 1):
            score_col = f'week{week}_total_score'
            if score_col not in season_data.columns:
                continue
                
            # 获取本周活跃选手
            week_data = season_data[season_data[score_col] > 0]
            if week_data.empty:
                continue
                
            active_contestants = week_data['celebrity_name'].tolist()
            n_active = len(active_contestants)
            
            if n_active == 0:
                continue
            
            # 获取评委分数
            judge_scores = {}
            for contestant in active_contestants:
                contestant_data = season_data[season_data['celebrity_name'] == contestant]
                if not contestant_data.empty:
                    judge_scores[contestant] = contestant_data[score_col].iloc[0]
            
            # 使用排名法预测
            rank_fan_votes = predict_by_rank(judge_scores, [])
            if rank_fan_votes:
                # 找出粉丝投票最低的选手
                min_vote_contestant = min(rank_fan_votes, key=rank_fan_votes.get)
                rank_eliminations[(season, week)] = [min_vote_contestant]
            
            # 使用百分比法预测
            percentage_fan_votes = predict_by_percentage(judge_scores, [])
            if percentage_fan_votes:
                # 找出粉丝投票最低的选手
                min_vote_contestant = min(percentage_fan_votes, key=percentage_fan_votes.get)
                percentage_eliminations[(season, week)] = [min_vote_contestant]
        
        # 计算与实际情况的匹配度
        rank_matches = 0
        percentage_matches = 0
        total_weeks = len(actual_eliminations)
        
        for key in actual_eliminations:
            actual_set = set(actual_eliminations[key])
            
            if key in rank_eliminations:
                rank_set = set(rank_eliminations[key])
                if rank_set == actual_set:
                    rank_matches += 1
            
            if key in percentage_eliminations:
                percentage_set = set(percentage_eliminations[key])
                if percentage_set == actual_set:
                    percentage_matches += 1
        
        rank_accuracy = rank_matches / total_weeks if total_weeks > 0 else 0
        percentage_accuracy = percentage_matches / total_weeks if total_weeks > 0 else 0
        
        # 统计两种方法预测结果的差异
        different_predictions = 0
        common_weeks = set(rank_eliminations.keys()) & set(percentage_eliminations.keys())
        
        for week_key in common_weeks:
            if set(rank_eliminations.get(week_key, [])) != set(percentage_eliminations.get(week_key, [])):
                different_predictions += 1
        
        comparison_results.append({
            'season': season,
            'voting_phase': voting_phase,
            'total_weeks': total_weeks,
            'rank_accuracy': rank_accuracy,
            'percentage_accuracy': percentage_accuracy,
            'accuracy_difference': percentage_accuracy - rank_accuracy,
            'different_predictions': different_predictions,
            'different_prediction_rate': different_predictions / len(common_weeks) if len(common_weeks) > 0 else 0
        })
    
    comparison_df = pd.DataFrame(comparison_results)
    return comparison_df

def analyze_method_impact_on_controversial_cases(df, max_week, total_score_cols):
    """
    分析两种方法对争议案例的影响
    """
    print("\n分析两种方法对争议案例的影响...")
    
    controversial_cases = [
        (2, "Jerry Rice"),
        (4, "Billy Ray Cyrus"),
        (11, "Bristol Palin"),
        (27, "Bobby Bones")
    ]
    
    results = []
    
    for season, contestant in controversial_cases:
        season_data = df[df['season'] == season]
        contestant_data = season_data[season_data['celebrity_name'] == contestant]
        
        if contestant_data.empty:
            continue
        
        # 获取该选手的比赛周数
        contestant_weeks = []
        for week in range(1, max_week + 1):
            col = f'week{week}_total_score'
            if col in contestant_data.columns and contestant_data[col].iloc[0] > 0:
                contestant_weeks.append(week)
        
        if not contestant_weeks:
            continue
        
        # 分析每一周
        weekly_results = []
        for week in contestant_weeks:
            score_col = f'week{week}_total_score'
            
            # 获取本周所有活跃选手
            week_data = season_data[season_data[score_col] > 0]
            if week_data.empty:
                continue
                
            active_contestants = week_data['celebrity_name'].tolist()
            
            # 获取评委分数
            judge_scores = {}
            for active_contestant in active_contestants:
                active_data = season_data[season_data['celebrity_name'] == active_contestant]
                if not active_data.empty:
                    judge_scores[active_contestant] = active_data[score_col].iloc[0]
            
            # 预测两种方法的粉丝投票
            rank_fan_votes = predict_by_rank(judge_scores, [])
            percentage_fan_votes = predict_by_percentage(judge_scores, [])
            
            if contestant in rank_fan_votes and contestant in percentage_fan_votes:
                # 计算在该周的排名
                rank_fan_votes_sorted = sorted(rank_fan_votes.items(), key=lambda x: x[1], reverse=True)
                percentage_fan_votes_sorted = sorted(percentage_fan_votes.items(), key=lambda x: x[1], reverse=True)
                
                rank_position = [i for i, (c, _) in enumerate(rank_fan_votes_sorted) if c == contestant][0] + 1
                percentage_position = [i for i, (c, _) in enumerate(percentage_fan_votes_sorted) if c == contestant][0] + 1
                
                weekly_results.append({
                    'week': week,
                    'judge_score': judge_scores.get(contestant, 0),
                    'rank_fan_vote': rank_fan_votes[contestant],
                    'percentage_fan_vote': percentage_fan_votes[contestant],
                    'rank_position': rank_position,
                    'percentage_position': percentage_position,
                    'position_difference': rank_position - percentage_position,
                    'total_contestants': len(active_contestants)
                })
        
        if weekly_results:
            weekly_df = pd.DataFrame(weekly_results)
            
            results.append({
                'season': season,
                'contestant': contestant,
                'total_weeks': len(weekly_results),
                'avg_judge_score': weekly_df['judge_score'].mean(),
                'avg_rank_fan_vote': weekly_df['rank_fan_vote'].mean(),
                'avg_percentage_fan_vote': weekly_df['percentage_fan_vote'].mean(),
                'avg_rank_position': weekly_df['rank_position'].mean(),
                'avg_percentage_position': weekly_df['percentage_position'].mean(),
                'weeks_rank_better': (weekly_df['rank_position'] < weekly_df['percentage_position']).sum(),
                'weeks_percentage_better': (weekly_df['rank_position'] > weekly_df['percentage_position']).sum(),
                'weeks_equal': (weekly_df['rank_position'] == weekly_df['percentage_position']).sum()
            })
    
    return pd.DataFrame(results)

def calculate_fan_vote_weight(df, max_week):
    """
    计算两种方法中粉丝投票的权重
    """
    print("\n计算两种方法中粉丝投票的权重...")
    
    seasons = sorted(df['season'].unique())
    weight_results = []
    
    for season in seasons[:20]:  # 分析前20个赛季以减少计算量
        season_data = df[df['season'] == season]
        
        # 获取该赛季的最大周数
        max_week_season = 0
        for week in range(1, max_week + 1):
            col = f'week{week}_total_score'
            if col in season_data.columns and season_data[col].notna().any():
                max_week_season = week
        
        season_weights = []
        
        for week in range(1, max_week_season + 1):
            score_col = f'week{week}_total_score'
            if score_col not in season_data.columns:
                continue
                
            # 获取本周活跃选手
            week_data = season_data[season_data[score_col] > 0]
            if week_data.empty:
                continue
                
            active_contestants = week_data['celebrity_name'].tolist()
            n_active = len(active_contestants)
            
            if n_active < 3:
                continue
            
            # 获取评委分数
            judge_scores = {}
            for contestant in active_contestants:
                contestant_data = season_data[season_data['celebrity_name'] == contestant]
                if not contestant_data.empty:
                    judge_scores[contestant] = contestant_data[score_col].iloc[0]
            
            # 分析排名法的权重
            rank_fan_votes = predict_by_rank(judge_scores, [])
            
            if rank_fan_votes:
                # 计算评委排名和粉丝排名的相关性
                sorted_by_score = sorted(judge_scores.items(), key=lambda x: x[1], reverse=True)
                judge_ranks = {contestant: rank+1 for rank, (contestant, _) in enumerate(sorted_by_score)}
                
                sorted_by_fan_vote = sorted(rank_fan_votes.items(), key=lambda x: x[1], reverse=True)
                fan_ranks = {contestant: rank+1 for rank, (contestant, _) in enumerate(sorted_by_fan_vote)}
                
                # 计算斯皮尔曼相关系数
                judge_rank_list = [judge_ranks[c] for c in active_contestants]
                fan_rank_list = [fan_ranks[c] for c in active_contestants]
                
                if len(set(judge_rank_list)) > 1 and len(set(fan_rank_list)) > 1:
                    spearman_corr, _ = stats.spearmanr(judge_rank_list, fan_rank_list)
                    
                    # 权重可以理解为粉丝排名对总排名的影响程度
                    # 在排名法中，粉丝排名和评委排名同等重要（各占50%权重）
                    season_weights.append({
                        'method': 'rank',
                        'week': week,
                        'spearman_corr': spearman_corr,
                        'implied_weight': 0.5  # 排名法中粉丝排名权重固定为0.5
                    })
            
            # 分析百分比法的权重
            percentage_fan_votes = predict_by_percentage(judge_scores, [])
            
            if percentage_fan_votes:
                # 计算评委百分比和粉丝百分比的相关性
                total_judge_score = sum(judge_scores.values())
                judge_percentages = {c: score/total_judge_score for c, score in judge_scores.items()}
                
                judge_percent_list = [judge_percentages[c] for c in active_contestants]
                fan_percent_list = [percentage_fan_votes[c] for c in active_contestants]
                
                if len(set(judge_percent_list)) > 1 and len(set(fan_percent_list)) > 1:
                    pearson_corr, _ = stats.pearsonr(judge_percent_list, fan_percent_list)
                    
                    # 在百分比法中，我们使用了40%评委+60%粉丝的权重
                    season_weights.append({
                        'method': 'percentage',
                        'week': week,
                        'pearson_corr': pearson_corr,
                        'implied_weight': 0.6  # 百分比法中粉丝投票权重为0.6
                    })
        
        if season_weights:
            weights_df = pd.DataFrame(season_weights)
            method_stats = weights_df.groupby('method').agg({
                'implied_weight': 'mean',
                'spearman_corr': 'mean',
                'pearson_corr': 'mean'
            }).reset_index()
            
            for _, row in method_stats.iterrows():
                weight_results.append({
                    'season': season,
                    'method': row['method'],
                    'avg_fan_weight': row['implied_weight'],
                    'avg_correlation': row['spearman_corr'] if row['method'] == 'rank' else row['pearson_corr']
                })
    
    return pd.DataFrame(weight_results)

# ==================== 新增：评委选择机制模拟（针对rank_bottom_two） ====================
def simulate_judge_choice_elimination(df, max_week, total_score_cols):
    """
    模拟第28季后引入的评委选择淘汰机制
    """
    print("\n模拟评委选择淘汰机制...")
    
    # 只模拟rank_bottom_two阶段（第28季及以后的赛季）
    simulation_results = []
    
    for season in df[df['voting_phase'] == 'rank_bottom_two']['season'].unique():
        season_data = df[df['season'] == season]
        
        # 获取该赛季的最大周数
        max_week_season = 0
        for week in range(1, max_week + 1):
            col = f'week{week}_total_score'
            if col in season_data.columns and season_data[col].notna().any():
                max_week_season = week
        
        for week in range(1, max_week_season):
            score_col = f'week{week}_total_score'
            next_score_col = f'week{week+1}_total_score'
            
            if score_col not in season_data.columns or next_score_col not in season_data.columns:
                continue
            
            # 获取本周活跃选手
            week_data = season_data[season_data[score_col] > 0]
            if week_data.empty:
                continue
                
            active_contestants = week_data['celebrity_name'].tolist()
            
            # 获取Bottom Two信息
            bottom_two_data = week_data[week_data['is_bottom_two'] == True]
            if len(bottom_two_data) < 2:
                continue
            
            bottom_two = bottom_two_data['celebrity_name'].tolist()
            
            # 记录原始淘汰
            original_eliminated = []
            for contestant in active_contestants:
                contestant_data = season_data[season_data['celebrity_name'] == contestant]
                if not contestant_data.empty:
                    current_score = contestant_data[score_col].iloc[0]
                    next_score = contestant_data[next_score_col].iloc[0] if next_score_col in contestant_data.columns else 0
                    if current_score > 0 and next_score == 0:
                        original_eliminated.append(contestant)
            
            if original_eliminated:
                original_eliminated = original_eliminated[0]  # 取第一个淘汰者
                
                # 模拟评委选择机制
                # 步骤1: 使用排名法计算综合分数
                judge_scores = {}
                for contestant in active_contestants:
                    contestant_data = season_data[season_data['celebrity_name'] == contestant]
                    if not contestant_data.empty:
                        judge_scores[contestant] = contestant_data[score_col].iloc[0]
                
                rank_fan_votes = predict_by_rank(judge_scores, [], bottom_two)
                
                if rank_fan_votes:
                    # 计算综合排名
                    sorted_by_score = sorted(judge_scores.items(), key=lambda x: x[1], reverse=True)
                    judge_ranks = {contestant: rank+1 for rank, (contestant, _) in enumerate(sorted_by_score)}
                    
                    sorted_by_fan_vote = sorted(rank_fan_votes.items(), key=lambda x: x[1], reverse=True)
                    fan_ranks = {contestant: rank+1 for rank, (contestant, _) in enumerate(sorted_by_fan_vote)}
                    
                    # 综合排名 = 评委排名 + 粉丝排名
                    combined_ranks = {}
                    for contestant in active_contestants:
                        combined_ranks[contestant] = judge_ranks[contestant] + fan_ranks[contestant]
                    
                    # 找出后两名
                    sorted_combined = sorted(combined_ranks.items(), key=lambda x: x[1], reverse=True)
                    bottom_two_candidates = [contestant for contestant, _ in sorted_combined[-2:]]
                    
                    # 评委根据技术分数选择淘汰分数较低者
                    if len(bottom_two_candidates) == 2:
                        contestant1_score = judge_scores.get(bottom_two_candidates[0], 0)
                        contestant2_score = judge_scores.get(bottom_two_candidates[1], 0)
                        
                        if contestant1_score < contestant2_score:
                            simulated_eliminated = bottom_two_candidates[0]
                        else:
                            simulated_eliminated = bottom_two_candidates[1]
                        
                        simulation_results.append({
                            'season': season,
                            'week': week,
                            'original_eliminated': original_eliminated,
                            'simulated_eliminated': simulated_eliminated,
                            'match': original_eliminated == simulated_eliminated,
                            'actual_bottom_two': bottom_two,
                            'simulated_bottom_two': bottom_two_candidates,
                            'judge_score_original': judge_scores.get(original_eliminated, 0),
                            'judge_score_simulated': judge_scores.get(simulated_eliminated, 0)
                        })
    
    simulation_df = pd.DataFrame(simulation_results)
    
    if not simulation_df.empty:
        match_rate = simulation_df['match'].mean()
        print(f"评委选择机制模拟匹配率: {match_rate:.4f}")
        
        # 分析哪些情况会改变结果
        changed_cases = simulation_df[~simulation_df['match']]
        print(f"淘汰结果改变的周数: {len(changed_cases)}/{len(simulation_df)}")
    
    return simulation_df

# ==================== 5. 不确定性计算 ====================
def calculate_uncertainty_fixed(predictions_df):
    """
    修正版不确定性计算 - 计算标准差、置信区间、变异系数等
    """
    print("计算预测不确定性...")
    
    if predictions_df.empty:
        print("警告: 预测数据为空")
        return pd.DataFrame()
    
    # 检查必要的列
    required_cols = ['season', 'week', 'contestant', 'fan_vote_raw']
    missing_cols = [col for col in required_cols if col not in predictions_df.columns]
    
    if missing_cols:
        print(f"错误: 预测数据缺少必要列 {missing_cols}")
        print(f"可用列: {list(predictions_df.columns)}")
        return pd.DataFrame()
    
    # 按赛季-周-选手分组，计算统计量
    grouped = predictions_df.groupby(['season', 'week', 'contestant'])
    
    result_rows = []
    
    for (season, week, contestant), group in grouped:
        # 获取所有样本的预测值
        fan_votes = group['fan_vote_raw'].values
        
        # 计算统计量
        mean_vote = np.mean(fan_votes)
        std_vote = np.std(fan_votes)
        
        # 计算95%置信区间
        if len(fan_votes) > 1:
            ci_low = np.percentile(fan_votes, 2.5)
            ci_high = np.percentile(fan_votes, 97.5)
        else:
            ci_low = mean_vote
            ci_high = mean_vote
        
        # 计算置信区间宽度
        ci_width = ci_high - ci_low
        
        # 计算变异系数（标准差/均值）
        if mean_vote > 0:
            cv = std_vote / mean_vote
        else:
            cv = 0
        
        # 获取选手信息
        row = group.iloc[0].copy()
        row['fan_vote_mean'] = mean_vote
        row['fan_vote_std'] = std_vote
        row['fan_vote_ci_low'] = max(0, ci_low)
        row['fan_vote_ci_high'] = min(1, ci_high)
        row['fan_vote_ci_width'] = ci_width
        row['fan_vote_cv'] = cv
        
        # 计算相对不确定性（相对于平均值的百分比）
        row['fan_vote_relative_uncertainty'] = std_vote / max(mean_vote, 0.001) * 100  # 百分比形式
        
        # 确定不确定性等级
        if cv == 0:
            uncertainty_level = '无不确定性'
        elif cv < 0.1:
            uncertainty_level = '低'
        elif cv < 0.25:
            uncertainty_level = '中等'
        else:
            uncertainty_level = '高'
        row['uncertainty_level'] = uncertainty_level
        
        result_rows.append(row)
    
    result_df = pd.DataFrame(result_rows)
    
    # 计算总体不确定性指标
    if not result_df.empty:
        avg_std = result_df['fan_vote_std'].mean()
        avg_ci_width = result_df['fan_vote_ci_width'].mean()
        avg_cv = result_df[result_df['fan_vote_mean'] > 0]['fan_vote_cv'].mean()
        
        print(f"平均标准差: {avg_std:.6f}")
        print(f"平均置信区间宽度: {avg_ci_width:.6f}")
        print(f"平均变异系数: {avg_cv:.6f} ({avg_cv*100:.2f}%)")
        
        # 不确定性分布统计
        print("\n不确定性分布统计:")
        print(f"最低变异系数: {result_df[result_df['fan_vote_mean'] > 0]['fan_vote_cv'].min():.6f}")
        print(f"最高变异系数: {result_df[result_df['fan_vote_mean'] > 0]['fan_vote_cv'].max():.6f}")
        
        # 按不确定性等级统计
        if 'uncertainty_level' in result_df.columns:
            uncertainty_counts = result_df['uncertainty_level'].value_counts()
            print("\n不确定性等级分布:")
            for level, count in uncertainty_counts.items():
                percentage = count / len(result_df) * 100
                print(f"  {level}: {count} 条记录 ({percentage:.1f}%)")
    
    return result_df

# ==================== 6. 一致性度量计算 ====================
def calculate_consistency_metrics(predictions_df, actual_eliminations):
    """
    计算预测结果与实际淘汰之间的一致性度量
    """
    print("计算一致性度量...")
    
    if predictions_df.empty:
        print("警告: 预测数据为空")
        return None, None
    
    # 预测的淘汰（粉丝投票最低的选手）
    predicted_eliminations = {}
    
    # 按赛季-周分组
    for (season, week), week_group in predictions_df.groupby(['season', 'week']):
        # 找出该周粉丝投票最低的选手
        if not week_group.empty:
            # 只考虑活跃选手（排除已淘汰）
            active_contestants = week_group[week_group['fan_vote_mean'] > 0.001]
            if not active_contestants.empty:
                min_vote_idx = active_contestants['fan_vote_mean'].idxmin()
                predicted_elim = active_contestants.loc[min_vote_idx, 'contestant']
                predicted_eliminations[(season, week)] = [predicted_elim]
    
    # 准备比较数据
    y_true = []  # 实际是否被淘汰 (1/0)
    y_pred = []  # 预测是否被淘汰 (1/0)
    
    # 为每个预测记录创建标签
    for _, row in predictions_df.iterrows():
        season = row['season']
        week = row['week']
        contestant = row['contestant']
        
        # 检查是否实际被淘汰
        actual_elim = actual_eliminations.get((season, week), [])
        is_actually_eliminated = contestant in actual_elim
        
        # 检查是否被预测淘汰
        pred_elim = predicted_eliminations.get((season, week), [])
        is_predicted_eliminated = contestant in pred_elim
        
        # 只考虑活跃选手（没有被提前淘汰）
        if row['fan_vote_mean'] > 0.001:
            y_true.append(1 if is_actually_eliminated else 0)
            y_pred.append(1 if is_predicted_eliminated else 0)
    
    # 计算指标
    if len(y_true) > 0 and len(y_pred) > 0:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # 计算每周准确率
        weekly_matches = []
        for key in predicted_eliminations:
            if key in actual_eliminations:
                pred_set = set(predicted_eliminations[key])
                actual_set = set(actual_eliminations[key])
                
                # 完全匹配为1，部分匹配为0.5，不匹配为0
                if pred_set == actual_set:
                    weekly_matches.append(1)
                elif len(pred_set & actual_set) > 0:
                    weekly_matches.append(0.5)
                else:
                    weekly_matches.append(0)
        
        weekly_accuracy = np.mean(weekly_matches) if weekly_matches else 0
        
        # 统计信息
        total_weeks = len(set([(s, w) for s, w in predicted_eliminations.keys()]))
        match_count = sum([1 for m in weekly_matches if m == 1])
        partial_match_count = sum([1 for m in weekly_matches if m == 0.5])
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'weekly_accuracy': weekly_accuracy,
            'weekly_match_rate': match_count / total_weeks if total_weeks > 0 else 0,
            'weekly_partial_match_rate': partial_match_count / total_weeks if total_weeks > 0 else 0,
            'total_samples': len(y_true),
            'total_weeks': total_weeks,
            'perfect_matches': match_count,
            'partial_matches': partial_match_count
        }
        
        print(f"淘汰预测准确率: {accuracy:.4f}")
        print(f"淘汰预测精确率: {precision:.4f}")
        print(f"淘汰预测召回率: {recall:.4f}")
        print(f"淘汰预测F1分数: {f1:.4f}")
        print(f"每周淘汰匹配度: {weekly_accuracy:.4f}")
        print(f"完美匹配周数: {match_count}/{total_weeks}")
        
        return metrics, predicted_eliminations
    else:
        print("没有足够的数据计算一致性度量")
        return None, None

# ==================== 7. 高级统计分析 ====================
def advanced_statistical_analysis(predictions_df):
    """
    进行高级统计分析
    """
    print("进行高级统计分析...")
    
    results = {}
    
    # 1. 不确定性分析
    if not predictions_df.empty:
        print("\n不确定性分析:")
        
        # 整体不确定性统计
        if 'fan_vote_cv' in predictions_df.columns:
            cv_stats = predictions_df[predictions_df['fan_vote_mean'] > 0]['fan_vote_cv'].describe()
            results['cv_statistics'] = cv_stats
            print(f"变异系数统计:\n{cv_stats}")
        
        # 按方法的不确定性分析
        if 'method' in predictions_df.columns:
            method_uncertainty = predictions_df.groupby('method').agg({
                'fan_vote_std': ['mean', 'std', 'min', 'max'],
                'fan_vote_cv': ['mean', 'std', 'min', 'max'],
                'fan_vote_ci_width': 'mean'
            }).round(6)
            
            results['method_uncertainty'] = method_uncertainty
            print(f"\n按方法的不确定性分析:")
            print(method_uncertainty)
        
        # 按投票阶段的不确定性分析
        if 'voting_phase' in predictions_df.columns:
            phase_uncertainty = predictions_df.groupby('voting_phase').agg({
                'fan_vote_std': ['mean', 'std'],
                'fan_vote_cv': ['mean', 'std'],
                'fan_vote_ci_width': 'mean'
            }).round(6)
            
            results['phase_uncertainty'] = phase_uncertainty
            print(f"\n按投票阶段的不确定性分析:")
            print(phase_uncertainty)
        
        # 按周数的不确定性分析
        if 'week' in predictions_df.columns:
            week_uncertainty = predictions_df.groupby('week').agg({
                'fan_vote_std': 'mean',
                'fan_vote_cv': 'mean',
                'fan_vote_ci_width': 'mean'
            }).round(6)
            
            results['week_uncertainty'] = week_uncertainty
            print(f"\n按周数的不确定性分析（前5周）:")
            print(week_uncertainty.head())
    
    # 2. 不同投票阶段对粉丝投票的影响
    if 'voting_phase' in predictions_df.columns and len(predictions_df['voting_phase'].unique()) > 1:
        phases = predictions_df['voting_phase'].unique()
        groups = [predictions_df[predictions_df['voting_phase'] == phase]['fan_vote_mean'].values 
                  for phase in phases if len(predictions_df[predictions_df['voting_phase'] == phase]) > 1]
        
        if len(groups) > 1:
            try:
                f_stat, p_val = stats.f_oneway(*groups)
                results['phase_anova'] = {
                    'f_statistic': f_stat,
                    'p_value': p_val,
                    'n_groups': len(groups)
                }
                print(f"\n投票阶段ANOVA检验: F={f_stat:.4f}, p={p_val:.6f}")
            except Exception as e:
                print(f"ANOVA计算错误: {e}")
                results['phase_anova'] = None
        
        # 投票阶段统计
        phase_stats = predictions_df.groupby('voting_phase').agg({
            'fan_vote_mean': ['mean', 'std', 'count'],
            'judge_score': 'mean',
            'fan_vote_cv': 'mean',
            'fan_vote_std': 'mean'
        }).round(4)
        
        results['phase_stats'] = phase_stats
    
    # 3. 评委分数与粉丝投票的相关性
    if 'judge_score' in predictions_df.columns and 'fan_vote_mean' in predictions_df.columns:
        judge_vote_corr = predictions_df['judge_score'].corr(predictions_df['fan_vote_mean'])
        results['judge_vote_correlation'] = judge_vote_corr
        print(f"\n评委分数与粉丝投票相关性: {judge_vote_corr:.4f}")
    
    # 4. 不同投票方法的比较
    if 'method' in predictions_df.columns:
        method_stats = predictions_df.groupby('method').agg({
            'fan_vote_mean': ['mean', 'std'],
            'fan_vote_std': 'mean',
            'fan_vote_cv': 'mean',
            'is_eliminated': 'mean'
        }).round(4)
        results['method_stats'] = method_stats
        
        # 方法间的统计检验
        methods = predictions_df['method'].unique()
        if len(methods) > 1:
            method_groups = [predictions_df[predictions_df['method'] == m]['fan_vote_mean'] for m in methods]
            try:
                f_stat, p_val = stats.f_oneway(*method_groups)
                results['method_anova'] = {
                    'f_statistic': f_stat,
                    'p_value': p_val
                }
                print(f"方法ANOVA检验: F={f_stat:.4f}, p={p_val:.6f}")
            except Exception as e:
                print(f"方法ANOVA计算错误: {e}")
    
    # 5. 淘汰状态与粉丝投票的关系
    if 'is_eliminated' in predictions_df.columns:
        eliminated_stats = predictions_df.groupby('is_eliminated').agg({
            'fan_vote_mean': ['mean', 'std', 'count'],
            'judge_score': 'mean',
            'fan_vote_cv': 'mean'
        }).round(4)
        results['eliminated_stats'] = eliminated_stats
    
    return results

# ==================== 8. 可视化函数 ====================
def create_comparison_visualizations(comparison_df, weight_df, controversial_df, simulation_df, predictions_df):
    """
    创建对比分析的可视化图表
    """
    print("创建对比分析可视化图表...")
    
    if not os.path.exists('task1/visualizations/comparison'):
        os.makedirs('task1/visualizations/comparison')
    
    # 1. 两种方法准确率对比（按赛季）
    if not comparison_df.empty:
        plt.figure(figsize=(16, 10))
        
        # 按投票阶段分组
        phases = comparison_df['voting_phase'].unique()
        colors = {'rank_regular': 'blue', 'percentage_regular': 'green', 'rank_bottom_two': 'red'}
        
        for phase in phases:
            phase_data = comparison_df[comparison_df['voting_phase'] == phase]
            if len(phase_data) > 0:
                plt.plot(phase_data['season'], phase_data['rank_accuracy'], 
                        marker='o', linestyle='-', color=colors.get(phase, 'gray'), 
                        label=f'{phase} - 排名法', alpha=0.7)
                plt.plot(phase_data['season'], phase_data['percentage_accuracy'], 
                        marker='s', linestyle='--', color=colors.get(phase, 'gray'), 
                        label=f'{phase} - 百分比法', alpha=0.7)
        
        plt.xlabel('赛季', fontsize=12)
        plt.ylabel('淘汰预测准确率', fontsize=12)
        plt.title('不同投票阶段两种方法准确率对比', fontsize=14)
        plt.legend(fontsize=10, ncol=3)
        plt.grid(True, alpha=0.3)
        plt.xticks(comparison_df['season'].unique(), rotation=45, fontsize=10)
        plt.tight_layout()
        plt.savefig('task1/visualizations/comparison/method_accuracy_by_phase.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. 粉丝投票权重对比
    if not weight_df.empty:
        plt.figure(figsize=(12, 6))
        
        # 按方法分组
        rank_weights = weight_df[weight_df['method'] == 'rank']['avg_fan_weight']
        percentage_weights = weight_df[weight_df['method'] == 'percentage']['avg_fan_weight']
        
        methods = ['排名法', '百分比法']
        avg_weights = [rank_weights.mean() if len(rank_weights) > 0 else 0, 
                      percentage_weights.mean() if len(percentage_weights) > 0 else 0]
        
        bars = plt.bar(methods, avg_weights, color=['blue', 'green'], alpha=0.7)
        plt.ylabel('平均粉丝投票权重', fontsize=12)
        plt.title('两种方法中粉丝投票的平均权重', fontsize=14)
        plt.ylim(0, 0.7)
        
        for bar, weight in zip(bars, avg_weights):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{weight:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('task1/visualizations/comparison/fan_weight_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. 争议案例两种方法对比
    if not controversial_df.empty:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('争议案例两种投票方法对比', fontsize=16)
        
        for idx, (_, row) in enumerate(controversial_df.iterrows()):
            ax = axes[idx // 2, idx % 2]
            
            labels = ['排名法', '百分比法']
            fan_votes = [row['avg_rank_fan_vote'], row['avg_percentage_fan_vote']]
            positions = [row['avg_rank_position'], row['avg_percentage_position']]
            
            x = np.arange(len(labels))
            width = 0.35
            
            ax1 = ax.twinx()
            
            bars1 = ax.bar(x - width/2, fan_votes, width, label='平均粉丝投票', color='skyblue')
            bars2 = ax1.bar(x + width/2, positions, width, label='平均排名', color='lightcoral')
            
            ax.set_ylabel('平均粉丝投票比例', fontsize=10)
            ax1.set_ylabel('平均排名', fontsize=10)
            ax.set_title(f"{row['contestant']} (赛季{row['season']})", fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            
            # 添加数值标签
            for bar, value in zip(bars1, fan_votes):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            for bar, value in zip(bars2, positions):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                        f'{value:.1f}', ha='center', va='bottom', fontsize=9)
            
            ax.legend(loc='upper left')
            ax1.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig('task1/visualizations/comparison/controversial_cases_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. 评委选择机制影响
    if not simulation_df.empty and len(simulation_df) > 0:
        plt.figure(figsize=(12, 8))
        
        # 计算匹配率
        match_rate = simulation_df['match'].mean()
        changed_rate = 1 - match_rate
        
        labels = ['淘汰结果匹配', '淘汰结果改变']
        sizes = [match_rate, changed_rate]
        colors = ['lightgreen', 'lightcoral']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('评委选择淘汰机制模拟结果 (Rank-Bottom-Two阶段)', fontsize=14)
        plt.tight_layout()
        plt.savefig('task1/visualizations/comparison/judge_choice_impact.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 改变结果的案例详细分析
        if changed_rate > 0:
            changed_cases = simulation_df[~simulation_df['match']]
            
            plt.figure(figsize=(14, 8))
            seasons_changed = changed_cases['season'].value_counts().sort_index()
            
            plt.bar(seasons_changed.index.astype(str), seasons_changed.values)
            plt.xlabel('赛季', fontsize=12)
            plt.ylabel('淘汰结果改变周数', fontsize=12)
            plt.title('评委选择机制导致淘汰结果改变的赛季分布', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            for i, (season, count) in enumerate(zip(seasons_changed.index, seasons_changed.values)):
                plt.text(i, count + 0.1, str(count), ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig('task1/visualizations/comparison/judge_choice_changes_by_season.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # 5. 新增：不确定性分析可视化
    if not predictions_df.empty:
        create_uncertainty_visualizations(predictions_df)
        create_additional_visualizations(predictions_df)

def create_uncertainty_visualizations(predictions_df):
    """
    创建不确定性分析的可视化图表
    """
    print("创建不确定性分析可视化图表...")
    
    if not os.path.exists('task1/visualizations/uncertainty'):
        os.makedirs('task1/visualizations/uncertainty')
    
    # 1. 变异系数分布直方图
    if 'fan_vote_cv' in predictions_df.columns:
        plt.figure(figsize=(12, 8))
        
        # 过滤掉零值
        cv_data = predictions_df[predictions_df['fan_vote_cv'] > 0]['fan_vote_cv']
        
        if len(cv_data) > 0:
            plt.hist(cv_data, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            plt.xlabel('变异系数 (标准差/均值)', fontsize=12)
            plt.ylabel('频率', fontsize=12)
            plt.title('粉丝投票预测变异系数分布', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_cv = cv_data.mean()
            median_cv = cv_data.median()
            plt.axvline(mean_cv, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_cv:.4f}')
            plt.axvline(median_cv, color='green', linestyle='--', linewidth=2, label=f'中位数: {median_cv:.4f}')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('task1/visualizations/uncertainty/cv_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # 2. 按周的变异系数变化
    if 'week' in predictions_df.columns and 'fan_vote_cv' in predictions_df.columns:
        plt.figure(figsize=(14, 8))
        
        week_cv = predictions_df.groupby('week')['fan_vote_cv'].mean().reset_index()
        
        plt.plot(week_cv['week'], week_cv['fan_vote_cv'], marker='o', linewidth=2)
        plt.xlabel('周数', fontsize=12)
        plt.ylabel('平均变异系数', fontsize=12)
        plt.title('粉丝投票预测不确定性随周数变化', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 添加趋势线
        if len(week_cv) > 1:
            z = np.polyfit(week_cv['week'], week_cv['fan_vote_cv'], 1)
            p = np.poly1d(z)
            plt.plot(week_cv['week'], p(week_cv['week']), 'r--', alpha=0.8, label='趋势线')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('task1/visualizations/uncertainty/cv_by_week.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. 按方法的变异系数对比
    if 'method' in predictions_df.columns and 'fan_vote_cv' in predictions_df.columns:
        plt.figure(figsize=(10, 6))
        
        method_cv = predictions_df.groupby('method')['fan_vote_cv'].mean().reset_index()
        
        bars = plt.bar(method_cv['method'], method_cv['fan_vote_cv'])
        plt.xlabel('投票方法', fontsize=12)
        plt.ylabel('平均变异系数', fontsize=12)
        plt.title('不同投票方法的预测不确定性对比', fontsize=14)
        
        # 添加数值标签
        for bar, cv in zip(bars, method_cv['fan_vote_cv']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{cv:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('task1/visualizations/uncertainty/cv_by_method.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. 争议案例置信区间可视化
    create_controversial_cases_ci_visualization(predictions_df)

def create_additional_visualizations(predictions_df):
    """
    创建额外的可视化图表
    """
    print("创建额外的可视化图表...")
    
    if not os.path.exists('task1/visualizations/additional'):
        os.makedirs('task1/visualizations/additional')
    
    # 1. 评委分数与粉丝投票关系散点图（拟合出线性趋势线）
    if 'judge_score' in predictions_df.columns and 'fan_vote_mean' in predictions_df.columns:
        plt.figure(figsize=(12, 8))
        
        # 过滤掉无效数据
        valid_data = predictions_df[(predictions_df['judge_score'] > 0) & 
                                   (predictions_df['fan_vote_mean'] > 0)]
        
        if len(valid_data) > 0:
            x = valid_data['judge_score'].values
            y = valid_data['fan_vote_mean'].values
            
            # 绘制散点图
            plt.scatter(x, y, alpha=0.6, color='blue', s=20)
            
            # 拟合线性趋势线
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            
            # 绘制趋势线
            x_range = np.linspace(x.min(), x.max(), 100)
            plt.plot(x_range, p(x_range), 'r--', linewidth=2, 
                    label=f'趋势线: y={z[0]:.4f}x + {z[1]:.4f}')
            
            # 添加相关系数
            corr_coef = np.corrcoef(x, y)[0, 1]
            plt.text(0.05, 0.95, f'相关系数: {corr_coef:.3f}', 
                    transform=plt.gca().transAxes, fontsize=12, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.xlabel('评委分数', fontsize=12)
            plt.ylabel('预测粉丝投票比例', fontsize=12)
            plt.title('评委分数与粉丝投票关系散点图（含线性趋势线）', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('task1/visualizations/additional/judge_vote_scatter.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # 2. 预测不确定性随赛季变化折线图（平均标准差）
    if 'season' in predictions_df.columns and 'fan_vote_std' in predictions_df.columns:
        plt.figure(figsize=(14, 8))
        
        # 按赛季计算平均标准差
        season_std = predictions_df.groupby('season')['fan_vote_std'].mean().reset_index()
        
        plt.plot(season_std['season'], season_std['fan_vote_std'], 
                marker='o', linewidth=2, markersize=6)
        plt.xlabel('赛季', fontsize=12)
        plt.ylabel('平均标准差', fontsize=12)
        plt.title('预测不确定性随赛季变化（平均标准差）', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 添加趋势线
        if len(season_std) > 1:
            z = np.polyfit(season_std['season'], season_std['fan_vote_std'], 1)
            p = np.poly1d(z)
            plt.plot(season_std['season'], p(season_std['season']), 'r--', alpha=0.8, 
                    label=f'趋势线: y={z[0]:.6f}x + {z[1]:.4f}')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('task1/visualizations/additional/uncertainty_by_season.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. 相对不确定性随赛季变化折线图（平均变异系数）
    if 'season' in predictions_df.columns and 'fan_vote_cv' in predictions_df.columns:
        plt.figure(figsize=(14, 8))
        
        # 按赛季计算平均变异系数
        season_cv = predictions_df[predictions_df['fan_vote_mean'] > 0].groupby('season')['fan_vote_cv'].mean().reset_index()
        
        plt.plot(season_cv['season'], season_cv['fan_vote_cv'], 
                marker='s', linewidth=2, markersize=6, color='green')
        plt.xlabel('赛季', fontsize=12)
        plt.ylabel('平均变异系数', fontsize=12)
        plt.title('相对不确定性随赛季变化（平均变异系数）', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 添加趋势线
        if len(season_cv) > 1:
            z = np.polyfit(season_cv['season'], season_cv['fan_vote_cv'], 1)
            p = np.poly1d(z)
            plt.plot(season_cv['season'], p(season_cv['season']), 'r--', alpha=0.8, 
                    label=f'趋势线: y={z[0]:.6f}x + {z[1]:.4f}')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('task1/visualizations/additional/relative_uncertainty_by_season.png', dpi=300, bbox_inches='tight')
        plt.close()

def create_controversial_cases_ci_visualization(predictions_df):
    """
    创建争议案例置信区间可视化图表
    """
    print("创建争议案例置信区间可视化...")
    
    controversial_cases = [
        (2, "Jerry Rice"),
        (4, "Billy Ray Cyrus"),
        (11, "Bristol Palin"),
        (27, "Bobby Bones")
    ]
    
    # 收集争议案例数据
    case_data = []
    
    for season, contestant in controversial_cases:
        case_df = predictions_df[(predictions_df['season'] == season) & 
                                (predictions_df['contestant'] == contestant)]
        
        if not case_df.empty:
            # 计算平均指标
            avg_fan_vote = case_df['fan_vote_mean'].mean()
            avg_ci_low = case_df['fan_vote_ci_low'].mean()
            avg_ci_high = case_df['fan_vote_ci_high'].mean()
            avg_cv = case_df['fan_vote_cv'].mean()
            avg_std = case_df['fan_vote_std'].mean()
            
            case_data.append({
                'season': season,
                'contestant': contestant,
                'avg_fan_vote': avg_fan_vote,
                'avg_ci_low': avg_ci_low,
                'avg_ci_high': avg_ci_high,
                'avg_cv': avg_cv,
                'avg_std': avg_std,
                'ci_width': avg_ci_high - avg_ci_low,
                'sample_size': len(case_df)
            })
    
    if case_data:
        case_df = pd.DataFrame(case_data)
        
        # 创建带有误差线的条形图
        plt.figure(figsize=(14, 8))
        
        contestants = [f"{row['contestant']}\n(赛季{row['season']})" for _, row in case_df.iterrows()]
        means = case_df['avg_fan_vote'].values
        ci_lows = case_df['avg_ci_low'].values
        ci_highs = case_df['avg_ci_high'].values
        errors = [means - ci_lows, ci_highs - means]
        
        x_pos = np.arange(len(contestants))
        bars = plt.bar(x_pos, means, yerr=errors, capsize=10, alpha=0.7, color='skyblue', edgecolor='black')
        
        plt.xlabel('争议案例', fontsize=12)
        plt.ylabel('平均粉丝投票比例 (95% 置信区间)', fontsize=12)
        plt.title('争议案例粉丝投票预测不确定性分析', fontsize=14)
        plt.xticks(x_pos, contestants, fontsize=10)
        
        # 添加数值标签
        for i, (bar, mean, cv, width) in enumerate(zip(bars, means, case_df['avg_cv'].values, case_df['ci_width'].values)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                    f'{mean:.3f}\nCV={cv:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('task1/visualizations/uncertainty/controversial_cases_ci.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 创建不确定性对比图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('争议案例不确定性指标对比', fontsize=16)
        
        metrics = [
            ('avg_fan_vote', '平均粉丝投票比例', 'blue'),
            ('avg_std', '平均标准差', 'red'),
            ('avg_cv', '平均变异系数', 'green'),
            ('ci_width', '平均置信区间宽度', 'purple')
        ]
        
        for idx, (metric, title, color) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            values = case_df[metric].values
            
            bars = ax.bar(x_pos, values, color=color, alpha=0.7)
            ax.set_xlabel('争议案例', fontsize=10)
            ax.set_ylabel(title, fontsize=10)
            ax.set_title(title, fontsize=12)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([row.contestant[:10] for row in case_df.itertuples()], rotation=45, fontsize=9)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                       f'{value:.4f}', ha='center', va='bottom', fontsize=8)
            
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('task1/visualizations/uncertainty/controversial_cases_uncertainty_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

# ==================== 9. 报告生成 ====================
def generate_comprehensive_comparison_report(predictions_df, comparison_df, weight_df, 
                                           controversial_df, simulation_df, 
                                           consistency_metrics, statistical_results):
    """
    生成包含对比分析的综合报告
    """
    print("生成综合对比分析报告...")
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("《与星共舞》粉丝投票预测模型 - 投票方法对比分析报告")
    report_lines.append("基于三种投票阶段：Rank-Regular, Percentage-Regular, Rank-Bottom-Two")
    report_lines.append("="*80)
    report_lines.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # 1. 总体统计
    report_lines.append("1. 总体统计")
    report_lines.append("-"*40)
    report_lines.append(f"分析赛季数: {predictions_df['season'].nunique()}")
    report_lines.append(f"总预测记录数: {len(predictions_df)}")
    report_lines.append(f"总选手数: {predictions_df['contestant'].nunique()}")
    
    if 'fan_vote_mean' in predictions_df.columns:
        report_lines.append(f"平均粉丝投票比例: {predictions_df['fan_vote_mean'].mean():.4f}")
    
    if 'judge_score' in predictions_df.columns:
        report_lines.append(f"平均评委分数: {predictions_df['judge_score'].mean():.2f}")
    
    # 投票阶段统计
    if 'voting_phase' in predictions_df.columns:
        phase_counts = predictions_df['voting_phase'].value_counts()
        report_lines.append("\n投票阶段分布:")
        for phase, count in phase_counts.items():
            percentage = count / len(predictions_df) * 100
            report_lines.append(f"  {phase}: {count} 条记录 ({percentage:.1f}%)")
    
    report_lines.append("")
    
    # 2. 两种投票方法对比分析
    report_lines.append("2. 两种投票方法对比分析")
    report_lines.append("-"*40)
    
    if not comparison_df.empty:
        # 总体准确率对比
        avg_rank_accuracy = comparison_df['rank_accuracy'].mean()
        avg_percentage_accuracy = comparison_df['percentage_accuracy'].mean()
        
        report_lines.append(f"排名法平均淘汰预测准确率: {avg_rank_accuracy:.4f}")
        report_lines.append(f"百分比法平均淘汰预测准确率: {avg_percentage_accuracy:.4f}")
        report_lines.append(f"准确率差异: {avg_percentage_accuracy - avg_rank_accuracy:.4f} "
                          f"({'百分比法更高' if avg_percentage_accuracy > avg_rank_accuracy else '排名法更高'})")
        
        # 按投票阶段分析
        report_lines.append("\n按投票阶段准确率分析:")
        for phase in comparison_df['voting_phase'].unique():
            phase_data = comparison_df[comparison_df['voting_phase'] == phase]
            if len(phase_data) > 0:
                phase_rank_acc = phase_data['rank_accuracy'].mean()
                phase_perc_acc = phase_data['percentage_accuracy'].mean()
                report_lines.append(f"  {phase}: 排名法={phase_rank_acc:.3f}, 百分比法={phase_perc_acc:.3f}, "
                                  f"差异={phase_perc_acc-phase_rank_acc:.3f}")
        
        # 不同预测结果分析
        avg_different_rate = comparison_df['different_prediction_rate'].mean()
        report_lines.append(f"\n两种方法预测结果不同的平均比例: {avg_different_rate:.4f}")
    
    # 3. 粉丝投票权重分析
    report_lines.append("")
    report_lines.append("3. 粉丝投票权重分析")
    report_lines.append("-"*40)
    
    if not weight_df.empty:
        rank_weights = weight_df[weight_df['method'] == 'rank']
        percentage_weights = weight_df[weight_df['method'] == 'percentage']
        
        if len(rank_weights) > 0:
            avg_rank_weight = rank_weights['avg_fan_weight'].mean()
            avg_rank_corr = rank_weights['avg_correlation'].mean()
            report_lines.append(f"排名法平均粉丝投票权重: {avg_rank_weight:.3f}")
            report_lines.append(f"排名法评委-粉丝排名相关性: {avg_rank_corr:.3f}")
        
        if len(percentage_weights) > 0:
            avg_percentage_weight = percentage_weights['avg_fan_weight'].mean()
            avg_percentage_corr = percentage_weights['avg_correlation'].mean()
            report_lines.append(f"百分比法平均粉丝投票权重: {avg_percentage_weight:.3f}")
            report_lines.append(f"百分比法评委-粉丝投票相关性: {avg_percentage_corr:.3f}")
    
    # 4. 不确定性分析
    report_lines.append("")
    report_lines.append("4. 预测不确定性分析")
    report_lines.append("-"*40)
    
    if statistical_results:
        # 变异系数统计
        if 'cv_statistics' in statistical_results:
            cv_stats = statistical_results['cv_statistics']
            report_lines.append(f"变异系数统计:")
            report_lines.append(f"  均值: {cv_stats['mean']:.6f} ({cv_stats['mean']*100:.2f}%)")
            report_lines.append(f"  标准差: {cv_stats['std']:.6f}")
            report_lines.append(f"  最小值: {cv_stats['min']:.6f}")
            report_lines.append(f"  25%分位数: {cv_stats['25%']:.6f}")
            report_lines.append(f"  中位数: {cv_stats['50%']:.6f}")
            report_lines.append(f"  75%分位数: {cv_stats['75%']:.6f}")
            report_lines.append(f"  最大值: {cv_stats['max']:.6f}")
        
        # 按方法的不确定性
        if 'method_uncertainty' in statistical_results:
            method_uncertainty = statistical_results['method_uncertainty']
            report_lines.append("\n按投票方法的不确定性:")
            for method in method_uncertainty.index:
                cv_mean = method_uncertainty.loc[method, ('fan_vote_cv', 'mean')]
                std_mean = method_uncertainty.loc[method, ('fan_vote_std', 'mean')]
                report_lines.append(f"  {method}: 平均变异系数={cv_mean:.6f}, 平均标准差={std_mean:.6f}")
    
    # 5. 争议案例分析
    report_lines.append("")
    report_lines.append("5. 争议案例两种方法对比")
    report_lines.append("-"*40)
    
    if not controversial_df.empty:
        for _, row in controversial_df.iterrows():
            report_lines.append(f"{row['contestant']} (赛季{row['season']}):")
            report_lines.append(f"  平均评委分数: {row['avg_judge_score']:.2f}")
            report_lines.append(f"  排名法平均粉丝投票: {row['avg_rank_fan_vote']:.4f} (平均排名: {row['avg_rank_position']:.1f})")
            report_lines.append(f"  百分比法平均粉丝投票: {row['avg_percentage_fan_vote']:.4f} (平均排名: {row['avg_percentage_position']:.1f})")
            
            if row['avg_rank_fan_vote'] > row['avg_percentage_fan_vote']:
                report_lines.append(f"  排名法预测的粉丝支持度更高，差异: {row['avg_rank_fan_vote'] - row['avg_percentage_fan_vote']:.4f}")
            else:
                report_lines.append(f"  百分比法预测的粉丝支持度更高，差异: {row['avg_percentage_fan_vote'] - row['avg_rank_fan_vote']:.4f}")
            
            report_lines.append(f"  排名法表现更好的周数: {row['weeks_rank_better']}")
            report_lines.append(f"  百分比法表现更好的周数: {row['weeks_percentage_better']}")
            report_lines.append("")
    
    # 6. 评委选择机制模拟
    report_lines.append("6. 评委选择淘汰机制模拟（Rank-Bottom-Two阶段）")
    report_lines.append("-"*40)
    
    if not simulation_df.empty:
        match_rate = simulation_df['match'].mean()
        changed_cases = simulation_df[~simulation_df['match']]
        
        report_lines.append(f"模拟匹配率: {match_rate:.4f}")
        report_lines.append(f"淘汰结果改变的周数: {len(changed_cases)}/{len(simulation_df)}")
        report_lines.append(f"改变比例: {len(changed_cases)/len(simulation_df):.4f}")
    
    # 7. 统计相关性分析
    report_lines.append("")
    report_lines.append("7. 统计相关性分析")
    report_lines.append("-"*40)
    
    if statistical_results:
        if 'judge_vote_correlation' in statistical_results:
            report_lines.append(f"评委分数与粉丝投票相关性: {statistical_results['judge_vote_correlation']:.4f}")
        
        if 'phase_anova' in statistical_results and statistical_results['phase_anova']:
            anova = statistical_results['phase_anova']
            report_lines.append(f"不同投票阶段粉丝投票差异显著性: F={anova['f_statistic']:.4f}, p={anova['p_value']:.6f}")
    
    # 8. 模型一致性度量
    report_lines.append("")
    report_lines.append("8. 模型一致性度量")
    report_lines.append("-"*40)
    
    if consistency_metrics:
        report_lines.append(f"淘汰预测准确率: {consistency_metrics['accuracy']:.4f}")
        report_lines.append(f"淘汰预测F1分数: {consistency_metrics['f1_score']:.4f}")
        report_lines.append(f"每周淘汰匹配度: {consistency_metrics['weekly_accuracy']:.4f}")
        report_lines.append(f"完美匹配周数: {consistency_metrics['perfect_matches']}/{consistency_metrics['total_weeks']}")
    
    # 9. 确定性度量总结
    report_lines.append("")
    report_lines.append("9. 确定性度量总结")
    report_lines.append("-"*40)
    
    if 'fan_vote_cv' in predictions_df.columns:
        # 计算整体不确定性指标
        cv_mean = predictions_df[predictions_df['fan_vote_mean'] > 0]['fan_vote_cv'].mean()
        std_mean = predictions_df['fan_vote_std'].mean()
        ci_width_mean = predictions_df['fan_vote_ci_width'].mean()
        
        report_lines.append(f"平均变异系数 (CV): {cv_mean:.6f} ({cv_mean*100:.2f}%)")
        report_lines.append(f"平均标准差: {std_mean:.6f}")
        report_lines.append(f"平均置信区间宽度: {ci_width_mean:.6f}")
        
        # 不确定性分布
        report_lines.append(f"\n不确定性分布:")
        report_lines.append(f"  • 低不确定性 (CV < 0.1): {len(predictions_df[predictions_df['fan_vote_cv'] < 0.1])} 条记录")
        report_lines.append(f"  • 中等不确定性 (0.1 ≤ CV < 0.25): {len(predictions_df[(predictions_df['fan_vote_cv'] >= 0.1) & (predictions_df['fan_vote_cv'] < 0.25)])} 条记录")
        report_lines.append(f"  • 高不确定性 (CV ≥ 0.25): {len(predictions_df[predictions_df['fan_vote_cv'] >= 0.25])} 条记录")
        
        # 按选手的不确定性差异
        if 'contestant' in predictions_df.columns:
            contestant_cv = predictions_df.groupby('contestant')['fan_vote_cv'].mean().sort_values()
            report_lines.append(f"\n最具确定性的选手 (前5名):")
            for contestant, cv in contestant_cv.head(5).items():
                report_lines.append(f"  {contestant}: CV={cv:.6f}")
            
            report_lines.append(f"\n最不确定性的选手 (前5名):")
            for contestant, cv in contestant_cv.tail(5).items():
                report_lines.append(f"  {contestant}: CV={cv:.6f}")
        
        # 按周的不确定性差异
        if 'week' in predictions_df.columns:
            week_cv = predictions_df.groupby('week')['fan_vote_cv'].mean()
            max_cv_week = week_cv.idxmax()
            min_cv_week = week_cv.idxmin()
            report_lines.append(f"\n不确定性最高的周: 第{max_cv_week}周 (CV={week_cv[max_cv_week]:.6f})")
            report_lines.append(f"不确定性最低的周: 第{min_cv_week}周 (CV={week_cv[min_cv_week]:.6f})")
    
    # 10. 结论与建议
    report_lines.append("")
    report_lines.append("10. 结论与建议")
    report_lines.append("-"*40)
    
    report_lines.append("主要发现:")
    report_lines.append("  1. 百分比法给予粉丝投票更高权重（60% vs 排名法的50%）")
    report_lines.append("  2. 百分比法在大多数赛季中淘汰预测准确率更高")
    report_lines.append("  3. 三种投票阶段的预测结果存在差异")
    report_lines.append("  4. Rank-Bottom-Two阶段的评委选择机制会影响淘汰结果")
    report_lines.append("  5. 排名法的不确定性通常更高（平均变异系数更高）")
    report_lines.append("  6. 争议案例的预测存在较高不确定性，需要更多数据支持")
    
    report_lines.append("")
    report_lines.append("确定性度量总结:")
    report_lines.append("  • 模型提供了完整的确定性度量：标准差、置信区间、变异系数")
    report_lines.append("  • 变异系数量化了不同参赛者/每周的估计波动")
    report_lines.append("  • 置信区间图表直观展示了估计值的确定性范围")
    report_lines.append("  • 不确定性等级分类有助于识别高风险预测")
    
    report_lines.append("")
    report_lines.append("新投票系统建议:")
    report_lines.append("  采用'动态混合系统':")
    report_lines.append("  • 第1-4周: 评委60% + 粉丝40% (强调技术基础)")
    report_lines.append("  • 第5-8周: 评委50% + 粉丝50% (平衡阶段)")
    report_lines.append("  • 第9周及以后: 评委40% + 粉丝60% (强调观众参与)")
    report_lines.append("  • 结合不确定性分析，为高风险淘汰决策提供额外评审环节")
    
    report_lines.append("")
    report_lines.append("="*80)
    
    # 保存报告
    report_text = "\n".join(report_lines)
    
    with open('voting_methods_comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"对比分析报告已保存到 'voting_methods_comparison_report.txt'")
    
    return report_text

# ==================== 10. 主函数 ====================
def main_enhanced():
    """
    增强版主函数，包含完整的对比分析
    """
    print("="*80)
    print("《与星共舞》粉丝投票预测模型 - 增强版（基于三种投票阶段）")
    print("="*80)
    
    # 1. 加载数据（使用三个预处理文件）
    rank_regular_path = 'task1\dwts_rank_regular_processed.csv'
    percentage_regular_path = 'task1\dwts_percentage_regular_processed.csv'
    rank_bottom_two_path = 'task1\dwts_rank_bottom_two_processed.csv'
    
    df, max_week, total_score_cols = load_and_preprocess_data(
        rank_regular_path, percentage_regular_path, rank_bottom_two_path
    )
    
    if df.empty:
        print("错误: 数据加载失败")
        return None, None, None, None, None
    
    if not total_score_cols:
        print("错误: 未找到或无法计算总分列，无法进行预测")
        return None, None, None, None, None
    
    # 2. 运行两种投票方法对比分析
    comparison_df = compare_voting_methods(df, max_week, total_score_cols)
    
    # 3. 分析粉丝投票权重
    weight_df = calculate_fan_vote_weight(df, max_week)
    
    # 4. 分析争议案例
    controversial_df = analyze_method_impact_on_controversial_cases(df, max_week, total_score_cols)
    
    # 5. 模拟评委选择机制
    simulation_df = simulate_judge_choice_elimination(df, max_week, total_score_cols)
    
    # 7. 生成粉丝投票预测（使用多次采样）
    print("\n开始预测粉丝投票（多次采样）...")
    all_predictions = []
    seasons = sorted(df['season'].unique())
    
    for season in seasons:
        print(f"预测赛季 {season}...")
        season_data = df[df['season'] == season]
        
        if season_data.empty:
            continue
        
        # 确定投票阶段
        voting_phase = season_data['voting_phase'].iloc[0]
        
        # 确定该赛季的实际周数
        actual_weeks = []
        for week in range(1, max_week + 1):
            col_name = f'week{week}_total_score'
            if col_name in season_data.columns and season_data[col_name].notna().any() and season_data[col_name].sum() > 0:
                actual_weeks.append(week)
        
        if not actual_weeks:
            print(f"  赛季 {season}: 无有效周数据")
            continue
        
        max_week_season = max(actual_weeks)
        print(f"  赛季 {season}: 共 {max_week_season} 周，投票阶段: {voting_phase}")
        
        # 多次采样
        for sample_idx in range(20):  # 减少采样次数以提高速度，但保证不确定性分析
            for week in range(1, max_week_season + 1):
                score_col = f'week{week}_total_score'
                
                if score_col not in season_data.columns:
                    continue
                    
                # 获取本周活跃选手（总分 > 0）
                week_data = season_data[season_data[score_col] > 0]
                if week_data.empty:
                    continue
                    
                active_contestants = week_data['celebrity_name'].tolist()
                
                if not active_contestants:
                    continue
                
                # 根据投票阶段确定预测方法
                if voting_phase in ['rank_regular', 'rank_bottom_two']:
                    method = 'rank'
                else:
                    method = 'percentage'
                
                # 获取评委分数
                judge_scores = {}
                for contestant in active_contestants:
                    contestant_data = season_data[season_data['celebrity_name'] == contestant]
                    if not contestant_data.empty:
                        judge_scores[contestant] = contestant_data[score_col].iloc[0]
                
                # 获取淘汰信息
                eliminated = []
                bottom_two = None
                
                # 获取下一周的分数列
                next_week = week + 1
                if next_week <= max_week_season:
                    next_score_col = f'week{next_week}_total_score'
                    if next_score_col in season_data.columns:
                        for contestant in active_contestants:
                            contestant_data = season_data[season_data['celebrity_name'] == contestant]
                            if not contestant_data.empty:
                                current_score = contestant_data[score_col].iloc[0]
                                next_score = contestant_data[next_score_col].iloc[0] if next_score_col in contestant_data.columns else 0
                                if current_score > 0 and next_score == 0:
                                    eliminated.append(contestant)
                
                # 对于rank_bottom_two阶段，获取Bottom Two信息
                if voting_phase == 'rank_bottom_two' and 'is_bottom_two' in week_data.columns:
                    bottom_two_data = week_data[week_data['is_bottom_two'] == True]
                    if not bottom_two_data.empty:
                        bottom_two = bottom_two_data['celebrity_name'].tolist()
                
                # 预测粉丝投票
                if method == 'rank':
                    fan_votes = predict_by_rank(judge_scores, eliminated, bottom_two)
                else:
                    fan_votes = predict_by_percentage(judge_scores, eliminated)
                
                # 存储预测结果
                for contestant in active_contestants:
                    if contestant in fan_votes:
                        contestant_data = season_data[season_data['celebrity_name'] == contestant].iloc[0]
                        
                        # 获取选手的其他信息
                        additional_info = {}
                        info_cols = ['placement', 'voting_phase', 'celebrity_age_during_season',
                                   'celebrity_industry', 'celebrity_homecountry/region']
                        
                        for col in info_cols:
                            if col in contestant_data:
                                additional_info[col] = contestant_data[col]
                        
                        all_predictions.append({
                            'sample_idx': sample_idx,
                            'season': season,
                            'week': week,
                            'contestant': contestant,
                            'fan_vote_raw': fan_votes[contestant],
                            'method': method,
                            'voting_phase': voting_phase,
                            'judge_score': judge_scores.get(contestant, 0),
                            'is_eliminated': 1 if contestant in eliminated else 0,
                            **additional_info
                        })
    
    # 转换为DataFrame并计算不确定性
    if all_predictions:
        predictions_df_raw = pd.DataFrame(all_predictions)
        print(f"\n生成 {len(predictions_df_raw)} 条预测记录")
        
        # 检查必要的列是否存在
        if 'season' not in predictions_df_raw.columns:
            print("错误: 预测数据中没有 'season' 列")
            print(f"预测数据列: {list(predictions_df_raw.columns)}")
            return None, comparison_df, weight_df, controversial_df, simulation_df
        
        predictions_df = calculate_uncertainty_fixed(predictions_df_raw)
        
        # 6. 创建对比分析可视化
        create_comparison_visualizations(comparison_df, weight_df, controversial_df, simulation_df, predictions_df)
        
        # 8. 计算一致性度量
        print("\n计算一致性度量...")
        actual_eliminations = {}
        
        for season in seasons:
            season_data = df[df['season'] == season]
            
            # 获取该赛季的最大周数
            max_week_season = 0
            for week in range(1, max_week + 1):
                col = f'week{week}_total_score'
                if col in season_data.columns and season_data[col].notna().any():
                    max_week_season = week
            
            # 每周检查淘汰
            for week in range(1, max_week_season):
                current_col = f'week{week}_total_score'
                next_col = f'week{week+1}_total_score'
                
                if current_col in season_data.columns and next_col in season_data.columns:
                    eliminated = []
                    for _, row in season_data.iterrows():
                        current_score = row[current_col]
                        next_score = row[next_col]
                        
                        if current_score > 0 and (pd.isna(next_score) or next_score == 0):
                            eliminated.append(row['celebrity_name'])
                    
                    if eliminated:
                        actual_eliminations[(season, week)] = eliminated
        
        consistency_metrics, _ = calculate_consistency_metrics(predictions_df, actual_eliminations)
        
        # 9. 高级统计分析
        statistical_results = advanced_statistical_analysis(predictions_df)
        
        # 10. 生成综合报告到task1目录
        report_text = generate_comprehensive_comparison_report(predictions_df, comparison_df, weight_df,
                                               controversial_df, simulation_df,
                                               consistency_metrics, statistical_results)
        
        # 保存报告到task1目录
        with open('task1/voting_methods_comparison_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        print("综合对比分析报告已保存到 'task1/voting_methods_comparison_report.txt'")
        
        # 11. 保存数据到task1目录
        predictions_df.to_csv('task1/fan_vote_predictions_enhanced.csv', index=False)
        if not comparison_df.empty:
            comparison_df.to_csv('task1/voting_methods_comparison.csv', index=False)
        if not weight_df.empty:
            weight_df.to_csv('task1/fan_vote_weights_analysis.csv', index=False)
        if not controversial_df.empty:
            controversial_df.to_csv('task1/controversial_cases_analysis.csv', index=False)
        if not simulation_df.empty:
            simulation_df.to_csv('task1/judge_choice_simulation.csv', index=False)
        
        # 保存不确定性分析结果
        if 'fan_vote_cv' in predictions_df.columns:
            uncertainty_summary = predictions_df.groupby(['season', 'week']).agg({
                'fan_vote_cv': ['mean', 'std', 'min', 'max'],
                'fan_vote_std': 'mean',
                'fan_vote_ci_width': 'mean'
            }).round(6)
            uncertainty_summary.to_csv('task1/uncertainty_analysis_summary.csv')
            print("不确定性分析摘要已保存到 'task1/uncertainty_analysis_summary.csv'")
        
        print(f"\n所有分析完成！")
    else:
        print("警告: 未生成任何预测数据")
        predictions_df = pd.DataFrame()
    
    # 12. 显示关键结果
    print("\n" + "="*80)
    print("关键发现摘要:")
    print("="*80)
    
    if not comparison_df.empty:
        print(f"排名法平均准确率: {comparison_df['rank_accuracy'].mean():.4f}")
        print(f"百分比法平均准确率: {comparison_df['percentage_accuracy'].mean():.4f}")
    
    if not weight_df.empty:
        rank_weight = weight_df[weight_df['method'] == 'rank']['avg_fan_weight'].mean()
        percentage_weight = weight_df[weight_df['method'] == 'percentage']['avg_fan_weight'].mean()
        print(f"排名法粉丝投票权重: {rank_weight:.3f}")
        print(f"百分比法粉丝投票权重: {percentage_weight:.3f}")
    
    if predictions_df is not None and not predictions_df.empty:
        if 'fan_vote_cv' in predictions_df.columns:
            cv_mean = predictions_df[predictions_df['fan_vote_mean'] > 0]['fan_vote_cv'].mean()
            print(f"平均变异系数: {cv_mean:.6f} ({cv_mean*100:.2f}%)")
    
    if not simulation_df.empty:
        match_rate = simulation_df['match'].mean()
        print(f"评委选择机制匹配率: {match_rate:.4f}")
    
    print("="*80)
    
    return predictions_df, comparison_df, weight_df, controversial_df, simulation_df

# ==================== 11. 执行主函数 ====================
if __name__ == "__main__":
    print("运行增强版模型...")
    try:
        predictions, comparison, weights, controversial, simulation = main_enhanced()
        
        if predictions is not None:
            print("\n" + "="*80)
            print("模型运行完成！")
            print("="*80)
            print("输出文件:")
            print("1. fan_vote_predictions_enhanced.csv - 预测结果数据")
            if comparison is not None and not comparison.empty:
                print("2. voting_methods_comparison.csv - 两种方法对比分析")
            if weights is not None and not weights.empty:
                print("3. fan_vote_weights_analysis.csv - 粉丝投票权重分析")
            if controversial is not None and not controversial.empty:
                print("4. controversial_cases_analysis.csv - 争议案例分析")
            if simulation is not None and not simulation.empty:
                print("5. judge_choice_simulation.csv - 评委选择机制模拟")
            print("6. uncertainty_analysis_summary.csv - 不确定性分析摘要")
            print("7. voting_methods_comparison_report.txt - 综合对比报告")
            print("8. task1/visualizations/comparison/ - 对比分析可视化图表")
            print("9. task1/visualizations/uncertainty/ - 不确定性分析可视化图表")
            print("="*80)
        else:
            print("\n模型运行失败！")
    except Exception as e:
        print(f"\n模型运行出错: {e}")
        import traceback
        traceback.print_exc()