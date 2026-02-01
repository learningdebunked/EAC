import pandas as pd

# Load the simulation results
results = pd.read_csv('simulation_results_test.csv')

# Group by protected_group and show counts + mean delta_spend
print('=== BY GROUP ===')
group_stats = results.groupby('protected_group').agg({
    'delta_spend': ['count', 'mean', 'sum']
}).round(4)
print(group_stats)

print('\n=== OVERALL ===')
print(f'Total observations: {len(results)}')
print(f'Total delta_spend sum: {results["delta_spend"].sum():.4f}')
print(f'Overall mean: {results["delta_spend"].mean():.4f}')

print('\n=== VERIFICATION ===')
# Manual weighted average calculation
for group in results['protected_group'].unique():
    g = results[results['protected_group'] == group]
    print(f'{group}: count={len(g)}, mean={g["delta_spend"].mean():.4f}, sum={g["delta_spend"].sum():.4f}')

total_sum = results['delta_spend'].sum()
total_count = len(results)
print(f'\nWeighted avg = {total_sum:.4f} / {total_count} = {total_sum/total_count:.4f}')