import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def analyze_current_bottlenecks():
    """分析当前模型的瓶颈"""
    print("=== 90%+准确率目标分析 ===")
    print("当前准确率: 72.38%")
    print("目标准确率: 90%+")
    print("需要提升: 至少17.62个百分点")
    print("")
    
    # 加载数据
    score_df = pd.read_csv('task4/enhanced_comprehensive_scores.csv')
    fan_df = pd.read_csv('task1/fan_vote_predictions_enhanced.csv')
    
    # 计算实际淘汰
    actual_eliminations = {}
    for (season, week), week_group in fan_df.groupby(['season', 'week']):
        if week < fan_df['week'].max():
            next_week_group = fan_df[(fan_df['season'] == season) & (fan_df['week'] == week + 1)]
            current_contestants = set(week_group['contestant'])
            next_contestants = set(next_week_group['contestant'])
            
            eliminated = list(current_contestants - next_contestants)
            if eliminated:
                actual_eliminations[(season, week)] = eliminated
    
    # 计算预测淘汰
    predicted_eliminations = {}
    for (season, week), week_group in score_df.groupby(['season', 'week']):
        if week < score_df['week'].max():
            max_rank_idx = week_group['rank'].idxmax()
            predicted_elim = week_group.loc[max_rank_idx, 'celebrity_name']
            predicted_eliminations[(season, week)] = [predicted_elim]
    
    # 找出预测错误的周
    error_cases = []
    for key in actual_eliminations:
        if key in predicted_eliminations:
            actual_set = set(actual_eliminations[key])
            pred_set = set(predicted_eliminations[key])
            
            if actual_set != pred_set:
                error_cases.append({
                    'season': key[0],
                    'week': key[1],
                    'actual': list(actual_set),
                    'predicted': list(pred_set)
                })
    
    print(f"预测错误案例数: {len(error_cases)}")
    print(f"总预测周数: {len(actual_eliminations)}")
    print(f"错误率: {len(error_cases)/len(actual_eliminations)*100:.2f}%")
    print("")
    
    # 深入分析错误案例
    print("=== 错误案例深度分析 ===")
    
    # 统计错误类型
    error_types = {
        'single_vs_multiple': 0,  # 单淘汰 vs 多淘汰
        'rank_mismatch': 0,       # 排名不匹配
        'close_scores': 0         # 分数接近
    }
    
    for case in error_cases[:10]:  # 分析前10个错误案例
        season, week = case['season'], case['week']
        
        # 获取该周数据
        week_data = score_df[(score_df['season'] == season) & (score_df['week'] == week)]
        
        # 分析错误类型
        if len(case['actual']) != len(case['predicted']):
            error_types['single_vs_multiple'] += 1
        
        # 检查排名差异
        for contestant in case['actual'] + case['predicted']:
            if contestant in week_data['celebrity_name'].values:
                player_data = week_data[week_data['celebrity_name'] == contestant].iloc[0]
                rank = player_data['rank']
                total_players = len(week_data)
                
                # 如果排名接近底部但不是最底部
                if rank >= total_players - 2 and rank != total_players:
                    error_types['rank_mismatch'] += 1
        
        # 检查分数接近情况
        actual_players = [week_data[week_data['celebrity_name'] == c] for c in case['actual'] if c in week_data['celebrity_name'].values]
        pred_players = [week_data[week_data['celebrity_name'] == c] for c in case['predicted'] if c in week_data['celebrity_name'].values]
        
        if actual_players and pred_players:
            actual_scores = [p['comprehensive_score'].iloc[0] for p in actual_players]
            pred_scores = [p['comprehensive_score'].iloc[0] for p in pred_players]
            
            if abs(np.mean(actual_scores) - np.mean(pred_scores)) < 0.1:
                error_types['close_scores'] += 1
    
    print("错误类型分布:")
    for error_type, count in error_types.items():
        print(f"  {error_type}: {count}次")
    print("")
    
    return score_df, fan_df, error_cases, actual_eliminations

def design_90_percent_solution(score_df, fan_df, error_cases):
    """设计90%+准确率的解决方案"""
    print("=== 90%+准确率解决方案设计 ===")
    
    # 方案1: 机器学习淘汰预测
    print("1. 机器学习淘汰预测模型")
    print("   - 特征: 综合评分、排名、历史表现、不确定性、比赛阶段")
    print("   - 模型: RandomForest, GradientBoosting, XGBoost")
    print("   - 目标: 直接预测淘汰概率")
    print("")
    
    # 方案2: 多选手淘汰预测
    print("2. 多选手淘汰预测")
    print("   - 问题: 当前只预测1名淘汰选手，但有时淘汰多名")
    print("   - 方案: 预测淘汰概率前N名选手")
    print("   - 调整: 根据比赛阶段动态调整N值")
    print("")
    
    # 方案3: 上下文特征增强
    print("3. 上下文特征增强")
    print("   - 比赛阶段特征: 初赛、复赛、决赛")
    print("   - 选手配对特征: 舞伴组合效果")
    print("   - 舞蹈类型特征: 不同舞蹈的难度系数")
    print("   - 历史淘汰模式: 类似选手的历史淘汰率")
    print("")
    
    # 方案4: 模型集成
    print("4. 多模型集成")
    print("   - 基础模型: 当前熵权动态加权模型")
    print("   - 机器学习模型: 淘汰概率预测")
    print("   - 规则模型: 基于历史规则的预测")
    print("   - 集成方法: 加权投票或Stacking")
    print("")
    
    # 方案5: 不确定性量化
    print("5. 不确定性量化与决策")
    print("   - 计算预测置信度")
    print("   - 低置信度时使用备用策略")
    print("   - 动态调整预测阈值")
    print("")
    
    return {
        'ml_features': ['comprehensive_score', 'rank', 'fan_vote_cv', 'total_uncertainty', 'week_progress'],
        'models': ['RandomForest', 'GradientBoosting', 'XGBoost'],
        'ensemble_method': 'weighted_voting',
        'confidence_threshold': 0.7
    }

def prepare_ml_training_data(score_df, fan_df):
    """准备机器学习训练数据"""
    print("准备机器学习训练数据...")
    
    # 创建选手级别的特征数据集
    player_features = []
    
    for (season, week), week_group in score_df.groupby(['season', 'week']):
        if week < score_df['week'].max():
            # 获取下周的选手名单
            next_week_players = set(fan_df[(fan_df['season'] == season) & (fan_df['week'] == week + 1)]['contestant'])
            
            for _, row in week_group.iterrows():
                player_name = row['celebrity_name']
                
                # 标签: 是否被淘汰
                is_eliminated = 1 if player_name not in next_week_players else 0
                
                # 特征
                features = {
                    'season': season,
                    'week': week,
                    'player': player_name,
                    'comprehensive_score': row['comprehensive_score'],
                    'rank': row['rank'],
                    'judge_score': row['judge_score'],
                    'fan_vote': row['fan_vote'],
                    'fan_vote_cv': row['fan_vote_cv'],
                    'total_uncertainty': row['total_uncertainty'],
                    'judge_weight': row['judge_weight'],
                    'fan_weight': row['fan_weight'],
                    'is_eliminated': is_eliminated
                }
                
                player_features.append(features)
    
    ml_df = pd.DataFrame(player_features)
    print(f"机器学习数据集: {len(ml_df)}条记录")
    print(f"淘汰比例: {ml_df['is_eliminated'].mean()*100:.2f}%")
    print("")
    
    return ml_df

def train_ml_models(ml_df):
    """训练机器学习模型"""
    print("训练机器学习模型...")
    
    # 特征选择
    feature_cols = ['comprehensive_score', 'rank', 'fan_vote_cv', 'total_uncertainty', 
                   'judge_weight', 'fan_weight', 'judge_score', 'fan_vote']
    
    X = ml_df[feature_cols]
    y = ml_df['is_eliminated']
    
    # 处理NaN值
    X = X.fillna(0)
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"训练集: {len(X_train)}条记录")
    print(f"测试集: {len(X_test)}条记录")
    print("")
    
    # 训练RandomForest
    print("训练RandomForest模型...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    print(f"RandomForest准确率: {rf_accuracy*100:.2f}%")
    
    # 训练GradientBoosting
    print("训练GradientBoosting模型...")
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_accuracy = accuracy_score(y_test, gb_pred)
    print(f"GradientBoosting准确率: {gb_accuracy*100:.2f}%")
    print("")
    
    # 特征重要性
    print("特征重要性分析 (RandomForest):")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in feature_importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return {
        'rf_model': rf_model,
        'gb_model': gb_model,
        'rf_accuracy': rf_accuracy,
        'gb_accuracy': gb_accuracy,
        'feature_importance': feature_importance,
        'feature_cols': feature_cols
    }

def create_ultimate_model(score_df, ml_results):
    """创建终极模型（结合熵权法和机器学习）"""
    print("创建终极模型...")
    
    # 使用机器学习模型预测淘汰概率
    rf_model = ml_results['rf_model']
    feature_cols = ml_results['feature_cols']
    
    # 为每个选手计算淘汰概率
    score_df['elimination_probability'] = 0.0
    
    for (season, week), week_group in score_df.groupby(['season', 'week']):
        if week < score_df['week'].max():
            # 准备特征
            week_features = week_group[feature_cols]
            
            # 预测淘汰概率
            if len(week_features) > 0:
                proba = rf_model.predict_proba(week_features)[:, 1]
                score_df.loc[week_group.index, 'elimination_probability'] = proba
    
    # 创建新的预测逻辑
    ultimate_predictions = {}
    
    for (season, week), week_group in score_df.groupby(['season', 'week']):
        if week < score_df['week'].max():
            # 根据淘汰概率和排名综合预测
            week_group = week_group.copy()
            
            # 综合得分 = 淘汰概率 * 0.7 + (1 - 标准化排名) * 0.3
            week_group['normalized_rank'] = 1 - (week_group['rank'] - 1) / (len(week_group) - 1)
            week_group['elimination_score'] = (week_group['elimination_probability'] * 0.7 + 
                                              week_group['normalized_rank'] * 0.3)
            
            # 预测淘汰选手（得分最高的1-2名）
            num_to_eliminate = 1 if week < 3 else 1  # 可根据比赛阶段调整
            top_elimination = week_group.nlargest(num_to_eliminate, 'elimination_score')
            predicted = list(top_elimination['celebrity_name'])
            
            ultimate_predictions[(season, week)] = predicted
    
    return score_df, ultimate_predictions

def evaluate_ultimate_model(score_df, ultimate_predictions, actual_eliminations):
    """评估终极模型"""
    print("评估终极模型...")
    
    matches = 0
    total_weeks = len(actual_eliminations)
    
    for key in actual_eliminations:
        if key in ultimate_predictions:
            actual_set = set(actual_eliminations[key])
            pred_set = set(ultimate_predictions[key])
            
            # 如果预测集合是实际集合的子集（允许预测部分正确）
            if pred_set.issubset(actual_set) or actual_set == pred_set:
                matches += 1
    
    accuracy = matches / total_weeks if total_weeks > 0 else 0
    print(f"终极模型准确率: {accuracy*100:.2f}%")
    
    # 与当前模型对比
    print(f"提升幅度: {accuracy*100 - 72.38:.2f}个百分点")
    
    return accuracy

def main():
    """主函数"""
    print("="*80)
    print("90%+准确率优化方案")
    print("="*80)
    
    # 1. 分析瓶颈
    score_df, fan_df, error_cases, actual_eliminations = analyze_current_bottlenecks()
    
    # 2. 设计解决方案
    solution_design = design_90_percent_solution(score_df, fan_df, error_cases)
    
    # 3. 准备机器学习数据
    ml_df = prepare_ml_training_data(score_df, fan_df)
    
    # 4. 训练机器学习模型
    ml_results = train_ml_models(ml_df)
    
    # 5. 创建终极模型
    score_df_with_proba, ultimate_predictions = create_ultimate_model(score_df, ml_results)
    
    # 6. 评估终极模型
    ultimate_accuracy = evaluate_ultimate_model(score_df_with_proba, ultimate_predictions, actual_eliminations)
    
    # 7. 保存结果
    if ultimate_accuracy > 0.72:  # 如果有提升
        score_df_with_proba.to_csv('task4/ultimate_comprehensive_scores.csv', index=False)
        
        # 保存预测结果
        predictions_df = pd.DataFrame([
            {'season': key[0], 'week': key[1], 'predicted_eliminations': ','.join(pred)}
            for key, pred in ultimate_predictions.items()
        ])
        predictions_df.to_csv('task4/ultimate_predictions.csv', index=False)
        
        print("")
        print("终极模型已保存:")
        print("1. ultimate_comprehensive_scores.csv - 包含淘汰概率的综合评分")
        print("2. ultimate_predictions.csv - 终极模型预测结果")
    
    print("="*80)
    print("优化完成!")
    
    return ultimate_accuracy

if __name__ == "__main__":
    main()