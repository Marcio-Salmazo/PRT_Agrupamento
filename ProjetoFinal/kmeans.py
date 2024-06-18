from distances import *

import random
import time

import pandas as pd


class KMeans:
    def __init__(self, data, labels, k=2, t=5, dist=euclidean_dist):
        self.k = k
        self.data = data
        self.labels = labels
        self.size = len(data)
        self.dist = dist
        self.t = t
        self.centroids = pd.DataFrame(columns=data.columns.values)
        self.clusters = [[] for _ in range(k)]

    def select_initial_centroids(self):
        random.seed(time.time_ns())

        for i in range(self.k):
            new_centroid_index = random.randint(0, self.size)

            for attr in self.data.columns.values:
                self.centroids.loc[i, attr] = self.data.loc[new_centroid_index, attr]


    def assign_clusters(self):
        clusters = [[] for _ in range(self.k)]
        for i in range(self.size):
            min_dist = float('inf')
            cluster = 0
            for c, centroid in self.centroids.iterrows():
                dist = self.dist(centroid, self.data.loc[i], self.data.columns.values)

                if dist < min_dist:
                    min_dist = dist
                    cluster = c

            clusters[cluster].append(i)

        self.clusters = clusters


    def calc_centroids(self):
        for i, clusters in enumerate(self.clusters):
            centroid = {}
            for attr in self.data.columns.values:
                centroid[attr] = 0

            for point in self.clusters[i]:
                for attr in self.data.columns.values:
                    centroid[attr] += self.data.loc[point][attr]

            for attr in self.data.columns.values:
                centroid[attr] = centroid[attr]/len(self.clusters[i])
            self.centroids.loc[i] = centroid

    def run(self):
        self.select_initial_centroids()

        for i in range(self.t):
            self.assign_clusters()
            self.calc_centroids()



def load_data(filename):
    return pd.read_csv(filename)


data_set = load_data("iris.csv")
labels = data_set["class"]
data_set.drop("class", axis=1, inplace=True)
kmeans = KMeans(data_set, labels, 3)
kmeans.run()
