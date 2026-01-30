import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('2026_MCM_Problem_C_Data.csv')

# Analyze the structure of judge scores
print('Analyzing judge score structure:')
print('Number of judges per week:')
for week in range(1, 12):
    week_cols = [f'week{week}_judge{i}_score' for i in range(1, 5)]
    available_judges = df[week_cols].notna().sum(axis=1)
    print(f'Week {week}: {available_judges.value_counts().to_dict()}')

# Check for seasons with different judge configurations
print('\nSeasons and their judge configurations:')
for season in sorted(df['season'].unique()):
    season_data = df[df['season'] == season]
    week1_judges = season_data['week1_judge4_score'].notna().sum()
    total_contestants = len(season_data)
    print(f'Season {season}: {total_contestants} contestants, {week1_judges} with 4 judges in week 1')

# Analyze elimination patterns
print('\nElimination patterns:')
elimination_counts = df['results'].value_counts()
print(elimination_counts)

# Check for controversial cases mentioned in the problem
controversial_cases = [
    ('Jerry Rice', 2),
    ('Billy Ray Cyrus', 4), 
    ('Bristol Palin', 11),
    ('Bobby Bones', 27)
]

print('\nControversial cases analysis:')
for name, season in controversial_cases:
    case_data = df[(df['celebrity_name'] == name) & (df['season'] == season)]
    if not case_data.empty:
        contestant = case_data.iloc[0]
        print(f'{name} (Season {season}):')
        print(f'  Placement: {contestant["placement"]}')
        print(f'  Results: {contestant["results"]}')
        # Get all judge scores for this contestant
        judge_cols = [col for col in df.columns if col.startswith('week') and col.endswith('_score')]
        scores = case_data[judge_cols].iloc[0]
        valid_scores = scores[scores.notna() & (scores > 0)]
        print(f'  Average judge score: {valid_scores.mean():.2f}')
        print(f'  Score range: {valid_scores.min():.1f} - {valid_scores.max():.1f}')
    else:
        print(f'{name} (Season {season}): Not found in data')