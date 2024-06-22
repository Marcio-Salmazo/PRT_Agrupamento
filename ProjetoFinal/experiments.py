import distances
from indexes import *
from sklearn.cluster import AffinityPropagation
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
                'cluster_amount': 5},
               {'data': 'processed_breast_cancer.csv',
                'cluster_amount': 2}]

algorithms = [kmeans.KMeans, ahc.AHC]

for algorithm in algorithms:
    for experiment in experiments:
        for i in range(10):
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

data_sets = ['glass.csv', 'user-knowledge.csv','processed-heart.csv','processed_breast_cancer.csv']

for i in range(10):
    for data_set in data_sets:
        data = pd.read_csv(data_set)
        labels = data["class"]
        data.drop("class", axis=1, inplace=True)

        alg = AffinityPropagation(random_state=time.time_ns() % 4294967295)

        start = time.time()
        alg.fit(data)
        finish = time.time()

        clusters = build_clusters(alg.labels_, len(alg.cluster_centers_))
        sil = silhouette(clusters, data, distances.euclidean_dist)
        p = purity(clusters, labels, labels.unique())

        df_dict['algorithm'].append('affinity')
        df_dict['data_set'].append(data_set)
        df_dict['k'].append(len(alg.cluster_centers_))
        df_dict['silhouette'].append(sil)
        df_dict['purity'].append(p)
        df_dict['duration'].append(finish-start)

df = pd.DataFrame(df_dict)
df.to_csv('experiment_result.csv', index=False)
