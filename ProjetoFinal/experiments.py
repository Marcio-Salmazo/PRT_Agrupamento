from indexes import *
import kmeans
import ahc
import pandas as pd
import time

df_dict = {'algorithm': [],
           'data_set': [],
           'k': [],
           'silhouette': [],
           'purity': [],
           'duration': []}

experiments = [{'data': 'glass.csv',
                'cluster_amount': 7},
               {'data': 'user-knowledge.csv',
                'cluster_amount': 5},
               {'data': 'processed-heart.csv',
                'cluster_amount': 5}]

algorithms = [kmeans.KMeans, ahc.AHC]

for algorithm in algorithms:
    for experiment in experiments:
        for i in range(2):
            data = pd.read_csv(experiment['data'])
            labels = data["class"]
            data.drop("class", axis=1, inplace=True)
            alg = algorithm(data, labels, experiment['cluster_amount'])

            start = time.time()
            alg.run()
            finish = time.time()

            sil = silhouette(alg.clusters, alg.data, alg.dist)
            p = purity(alg.clusters, alg.labels, alg.labels.unique())

            df_dict['algorithm'].append(alg.name)
            df_dict['data_set'].append(experiment['data'])
            df_dict['k'].append(experiment['cluster_amount'])
            df_dict['silhouette'].append(sil)
            df_dict['purity'].append(p)
            df_dict['duration'].append(finish-start)

df = pd.DataFrame(df_dict)
df.to_csv('experiment_result.csv')
