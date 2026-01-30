import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def load_and_preprocess_data():
    """Load and preprocess the DWTS data"""
    df = pd.read_csv('2026_MCM_Problem_C_Data.csv')
    
    # Calculate total judge scores for each contestant
    def calculate_weekly_scores(row, week):
        judge_cols = [f'week{week}_judge{i}_score' for i in range(1, 5)]
        scores = row[judge_cols]
        valid_scores = scores[(scores.notna()) & (scores > 0)]
        return valid_scores.sum() if len(valid_scores) > 0 else 0
    
    # Calculate total judge scores for each week
    for week in range(1, 12):
        df[f'week{week}_judge_total'] = df.apply(lambda row: calculate_weekly_scores(row, week), axis=1)
    
    # Calculate average judge score per contestant
    def calculate_total_score(row):
        judge_cols = [col for col in df.columns if col.startswith('week') and 'judge_total' in col]
        scores = []
        for col in judge_cols:
            if pd.notna(row[col]) and row[col] > 0:
                scores.append(row[col])
        return np.mean(scores) if scores else 0
    
    df['avg_judge_score'] = df.apply(calculate_total_score, axis=1)
    
    return df

def current_rank_method(df, week):
    """Implement the current rank-based voting method"""
    week_data = df[df[f'week{week}_judge_total'] > 0].copy()
    
    if len(week_data) == 0:
        return {}
    
    # Calculate judge ranks (lower score = higher rank)
    week_data['judge_rank'] = week_data[f'week{week}_judge_total'].rank(method='min')
    
    # For rank method, we need to estimate fan votes such that the final ranking
    # matches the actual elimination order
    results_order = []
    for _, row in week_data.iterrows():
        if 'Eliminated' in row['results'] and f'Week {week}' in row['results']:
            results_order.append((row['celebrity_name'], row['season'], 'eliminated'))
        elif row['placement'] <= 3:  # Made it to finals
            results_order.append((row['celebrity_name'], row['season'], 'finalist'))
        else:
            results_order.append((row['celebrity_name'], row['season'], 'safe'))
    
    return results_order

def current_percentage_method(df, week):
    """Implement the current percentage-based voting method"""
    week_data = df[df[f'week{week}_judge_total'] > 0].copy()
    
    if len(week_data) == 0:
        return {}
    
    # Calculate judge percentages
    total_judge_score = week_data[f'week{week}_judge_total'].sum()
    week_data['judge_percentage'] = week_data[f'week{week}_judge_total'] / total_judge_score
    
    return week_data

def proposed_weighted_method(df, week, judge_weight=0.4, fan_weight=0.6):
    """
    Proposed weighted voting system that balances judge expertise with fan engagement
    """
    week_data = df[df[f'week{week}_judge_total'] > 0].copy()
    
    if len(week_data) == 0:
        return {}
    
    # Calculate judge score (normalize to 0-100 scale)
    max_judge_score = week_data[f'week{week}_judge_total'].max()
    min_judge_score = week_data[f'week{week}_judge_total'].min()
    
    if max_judge_score == min_judge_score:
        week_data['judge_normalized'] = 50  # Equal scores
    else:
        week_data['judge_normalized'] = (
            (week_data[f'week{week}_judge_total'] - min_judge_score) / 
            (max_judge_score - min_judge_score) * 100
        )
    
    # Estimate fan votes using optimization (similar to previous model)
    active_contestants = week_data['celebrity_name'].tolist()
    active_seasons = week_data['season'].tolist()
    initial_votes = np.ones(len(active_contestants))
    
    def objective_function(fan_votes):
        """Objective function to minimize difference from actual results"""
        total_scores = []
        
        for i, (name, season) in enumerate(zip(active_contestants, active_seasons)):
            contestant_data = week_data[(week_data['celebrity_name'] == name) & 
                                       (week_data['season'] == season)]
            if len(contestant_data) == 0:
                continue
                
            contestant = contestant_data.iloc[0]
            judge_score = contestant['judge_normalized']
            
            # Convert fan votes to percentage
            total_fan_votes = sum(fan_votes)
            fan_percentage = fan_votes[i] / total_fan_votes if total_fan_votes > 0 else 0
            
            # Calculate weighted score
            weighted_score = (judge_weight * judge_score) + (fan_weight * fan_percentage * 100)
            
            total_scores.append((name, season, weighted_score))
        
        # Sort by total score (higher is better)
        total_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Calculate penalty based on how well this matches actual elimination order
        penalty = 0
        for i, (name, season, score) in enumerate(total_scores):
            contestant_data = week_data[(week_data['celebrity_name'] == name) & 
                                       (week_data['season'] == season)]
            if len(contestant_data) == 0:
                continue
                
            contestant = contestant_data.iloc[0]
            if 'Eliminated' in contestant['results'] and f'Week {week}' in contestant['results']:
                # This contestant should be eliminated (should have low score)
                expected_rank = len(active_contestants) - 1  # Last place
                penalty += abs(i - expected_rank)
        
        return penalty
    
    # Optimize fan vote estimates
    result = minimize(objective_function, initial_votes, method='Nelder-Mead')
    
    # Calculate final weighted scores
    final_scores = []
    for i, (name, season) in enumerate(zip(active_contestants, active_seasons)):
        contestant_data = week_data[(week_data['celebrity_name'] == name) & 
                                   (week_data['season'] == season)]
        if len(contestant_data) == 0:
            continue
            
        contestant = contestant_data.iloc[0]
        judge_score = contestant['judge_normalized']
        fan_vote = result.x[i]
        total_fan_votes = sum(result.x)
        fan_percentage = fan_vote / total_fan_votes if total_fan_votes > 0 else 0
        
        weighted_score = (judge_weight * judge_score) + (fan_weight * fan_percentage * 100)
        final_scores.append((name, season, weighted_score, judge_score, fan_percentage * 100))
    
    # Sort by weighted score
    final_scores.sort(key=lambda x: x[2], reverse=True)
    
    return final_scores

def proposed_hybrid_method(df, week):
    """
    Proposed hybrid method that uses different weights based on contestant performance
    """
    week_data = df[df[f'week{week}_judge_total'] > 0].copy()
    
    if len(week_data) == 0:
        return {}
    
    # Calculate judge score (normalize to 0-100 scale)
    max_judge_score = week_data[f'week{week}_judge_total'].max()
    min_judge_score = week_data[f'week{week}_judge_total'].min()
    
    if max_judge_score == min_judge_score:
        week_data['judge_normalized'] = 50  # Equal scores
    else:
        week_data['judge_normalized'] = (
            (week_data[f'week{week}_judge_total'] - min_judge_score) / 
            (max_judge_score - min_judge_score) * 100
        )
    
    # Estimate fan votes
    active_contestants = week_data['celebrity_name'].tolist()
    active_seasons = week_data['season'].tolist()
    initial_votes = np.ones(len(active_contestants))
    
    def objective_function(fan_votes):
        """Objective function to minimize difference from actual results"""
        total_scores = []
        
        for i, (name, season) in enumerate(zip(active_contestants, active_seasons)):
            contestant_data = week_data[(week_data['celebrity_name'] == name) & 
                                       (week_data['season'] == season)]
            if len(contestant_data) == 0:
                continue
                
            contestant = contestant_data.iloc[0]
            judge_score = contestant['judge_normalized']
            fan_vote = fan_votes[i]
            total_fan_votes = sum(fan_votes)
            fan_percentage = fan_vote / total_fan_votes if total_fan_votes > 0 else 0
            
            # Adaptive weighting: higher judge weight for higher judge scores
            if judge_score >= 80:  # Top performers get more judge weight
                judge_weight = 0.7
                fan_weight = 0.3
            elif judge_score >= 60:  # Middle performers get balanced weight
                judge_weight = 0.5
                fan_weight = 0.5
            else:  # Lower performers get more fan weight
                judge_weight = 0.3
                fan_weight = 0.7
            
            weighted_score = (judge_weight * judge_score) + (fan_weight * fan_percentage * 100)
            total_scores.append((name, season, weighted_score))
        
        # Sort by total score (higher is better)
        total_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Calculate penalty
        penalty = 0
        for i, (name, season, score) in enumerate(total_scores):
            contestant_data = week_data[(week_data['celebrity_name'] == name) & 
                                       (week_data['season'] == season)]
            if len(contestant_data) == 0:
                continue
                
            contestant = contestant_data.iloc[0]
            if 'Eliminated' in contestant['results'] and f'Week {week}' in contestant['results']:
                expected_rank = len(active_contestants) - 1
                penalty += abs(i - expected_rank)
        
        return penalty
    
    # Optimize fan vote estimates
    result = minimize(objective_function, initial_votes, method='Nelder-Mead')
    
    # Calculate final scores with adaptive weights
    final_scores = []
    for i, (name, season) in enumerate(zip(active_contestants, active_seasons)):
        contestant_data = week_data[(week_data['celebrity_name'] == name) & 
                                   (week_data['season'] == season)]
        if len(contestant_data) == 0:
            continue
            
        contestant = contestant_data.iloc[0]
        judge_score = contestant['judge_normalized']
        fan_vote = result.x[i]
        total_fan_votes = sum(result.x)
        fan_percentage = fan_vote / total_fan_votes if total_fan_votes > 0 else 0
        
        # Adaptive weighting
        if judge_score >= 80:
            judge_weight = 0.7
            fan_weight = 0.3
        elif judge_score >= 60:
            judge_weight = 0.5
            fan_weight = 0.5
        else:
            judge_weight = 0.3
            fan_weight = 0.7
        
        weighted_score = (judge_weight * judge_score) + (fan_weight * fan_percentage * 100)
        final_scores.append((name, season, weighted_score, judge_score, fan_percentage * 100))
    
    # Sort by weighted score
    final_scores.sort(key=lambda x: x[2], reverse=True)
    
    return final_scores

def compare_voting_methods():
    """Compare the performance of different voting methods"""
    df = load_and_preprocess_data()
    
    print("Comparing Voting Methods")
    print("=" * 50)
    
    # Test on a few representative seasons
    test_seasons = [5, 15, 25, 30]  # Mix of different eras
    
    for season in test_seasons:
        season_data = df[df['season'] == season]
        max_week = season_data['placement'].max()
        
        print(f"\nSeason {season} Analysis:")
        print("-" * 30)
        
        # Determine original voting method
        if season <= 2 or season >= 28:
            original_method = "Rank-based"
        else:
            original_method = "Percentage-based"
        
        print(f"Original method: {original_method}")
        
        # Test each week
        for week in range(1, min(max_week + 1, 6)):  # Test first 5 weeks
            week_data = season_data[season_data[f'week{week}_judge_total'] > 0]
            if len(week_data) == 0:
                continue
            
            print(f"  Week {week}: {len(week_data)} contestants")
            
            # Test proposed methods
            weighted_scores = proposed_weighted_method(season_data, week)
            hybrid_scores = proposed_hybrid_method(season_data, week)
            
            # Calculate agreement with actual eliminations
            actual_eliminations = []
            for _, contestant in week_data.iterrows():
                if 'Eliminated' in contestant['results'] and f'Week {week}' in contestant['results']:
                    actual_eliminations.append((contestant['celebrity_name'], contestant['season']))
            
            # Check agreement for weighted method
            weighted_eliminated = weighted_scores[-len(actual_eliminations):] if actual_eliminations else []
            weighted_agreement = 0
            for eliminated in actual_eliminations:
                if eliminated in [(score[0], score[1]) for score in weighted_eliminated]:
                    weighted_agreement += 1
            
            # Check agreement for hybrid method
            hybrid_eliminated = hybrid_scores[-len(actual_eliminations):] if actual_eliminations else []
            hybrid_agreement = 0
            for eliminated in actual_eliminations:
                if eliminated in [(score[0], score[1]) for score in hybrid_eliminated]:
                    hybrid_agreement += 1
            
            if actual_eliminations:
                print(f"    Weighted method agreement: {weighted_agreement}/{len(actual_eliminations)}")
                print(f"    Hybrid method agreement: {hybrid_agreement}/{len(actual_eliminations)}")

def analyze_fairness_metrics():
    """Analyze fairness metrics for different voting systems"""
    df = load_and_preprocess_data()
    
    print("\n\nFairness Analysis")
    print("=" * 50)
    
    # Analyze judge score distributions
    judge_scores = df['avg_judge_score'].dropna()
    
    print("Judge Score Distribution:")
    print(f"  Mean: {judge_scores.mean():.2f}")
    print(f"  Std Dev: {judge_scores.std():.2f}")
    print(f"  Min: {judge_scores.min():.2f}")
    print(f"  Max: {judge_scores.max():.2f}")
    
    # Analyze placement vs judge scores
    correlation = np.corrcoef(df['avg_judge_score'].dropna(), df['placement'].dropna())[0,1]
    print(f"\nJudge Score vs Placement Correlation: {correlation:.3f}")
    
    # Analyze by professional dancer
    pro_impact = df.groupby('ballroom_partner').agg({
        'avg_judge_score': 'mean',
        'placement': 'mean',
        'celebrity_name': 'count'
    }).round(2)
    
    pro_impact = pro_impact[pro_impact['celebrity_name'] >= 3]  # Only pros with 3+ contestants
    pro_impact = pro_impact.sort_values('avg_judge_score', ascending=False)
    
    print(f"\nProfessional Dancer Impact (min 3 contestants):")
    print(f"  Best average judge score: {pro_impact.iloc[0]['avg_judge_score']:.2f}")
    print(f"  Worst average judge score: {pro_impact.iloc[-1]['avg_judge_score']:.2f}")
    print(f"  Score difference: {pro_impact.iloc[0]['avg_judge_score'] - pro_impact.iloc[-1]['avg_judge_score']:.2f}")
    
    # Analyze by contestant characteristics
    industry_impact = df.groupby('celebrity_industry').agg({
        'avg_judge_score': 'mean',
        'placement': 'mean',
        'celebrity_name': 'count'
    }).round(2)
    
    industry_impact = industry_impact[industry_impact['celebrity_name'] >= 3]
    industry_impact = industry_impact.sort_values('avg_judge_score', ascending=False)
    
    print(f"\nIndustry Impact (min 3 contestants):")
    print(f"  Best average judge score: {industry_impact.iloc[0]['avg_judge_score']:.2f}")
    print(f"  Worst average judge score: {industry_impact.iloc[-1]['avg_judge_score']:.2f}")
    print(f"  Score difference: {industry_impact.iloc[0]['avg_judge_score'] - industry_impact.iloc[-1]['avg_judge_score']:.2f}")

def create_comparison_visualizations():
    """Create visualizations comparing voting methods"""
    print("\n\nCreating Comparison Visualizations...")
    
    # This would create charts comparing the different methods
    # For now, we'll just print a summary
    
    print("Visualization summary:")
    print("1. Judge score distributions by voting method")
    print("2. Elimination prediction accuracy comparison")
    print("3. Fairness metrics comparison")
    print("4. Professional dancer impact analysis")

def main():
    """Main analysis function"""
    compare_voting_methods()
    analyze_fairness_metrics()
    create_comparison_visualizations()
    
    print("\n\nRecommendations for DWTS Producers:")
    print("=" * 50)
    print("1. Implement a weighted voting system (40% judges, 60% fans)")
    print("2. Consider adaptive weighting based on performance quality")
    print("3. Maintain transparency in the voting process")
    print("4. Consider implementing a 'judges' save' mechanism for close calls")
    print("5. Regularly review and adjust the voting system based on data")
    print("6. Ensure all contestants have equal opportunities regardless of background")

if __name__ == "__main__":
    main()