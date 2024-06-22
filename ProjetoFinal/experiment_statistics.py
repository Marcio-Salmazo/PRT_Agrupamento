import pandas as pd
import numpy as np

stats = {'algorithm': [],
         'data_set': [],
         'k': [],
         'silhouette_mean': [],
         'silhouette_std': [],
         'purity_mean': [],
         'purity_std': [],
         'duration_mean': [],
         'duration_std': []}

df = pd.read_csv('experiment_result.csv')

algorithms = ['kmeans', 'ahc', 'affinity']
data_sets = ['glass.csv', 'processed-heart.csv', 'user-knowledge.csv', 'processed_breast_cancer.csv']

for alg in algorithms:
    for data_set in data_sets:
        alg_stat = {}
        silhouette = df[df['algorithm'].isin([alg])][df['data_set'].isin([data_set])]['silhouette']
        purity = df[df['algorithm'].isin([alg])][df['data_set'].isin([data_set])]['purity']
        duration = df[df['algorithm'].isin([alg])][df['data_set'].isin([data_set])]['duration']
        k = df[df['algorithm'].isin([alg])][df['data_set'].isin([data_set])]['k']

        stats['algorithm'].append(alg)
        stats['data_set'].append(data_set)
        stats['k'].append(np.mean(k))
        stats['silhouette_mean'].append(np.mean(silhouette))
        stats['silhouette_std'].append(np.std(silhouette))
        stats['purity_mean'].append(np.mean(purity))
        stats['purity_std'].append(np.std(purity))
        stats['duration_mean'].append(np.mean(duration))
        stats['duration_std'].append(np.std(duration))

df_statistics = pd.DataFrame(stats)
df_statistics.to_csv('experiments_statistics.csv')

