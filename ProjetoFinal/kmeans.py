from distances import *

import random
import time

import pandas as pd


class KMeans:
    def __init__(self, data, labels, k=2, t=100, dist=euclidean_dist):
        self.k = k
        self.data = data
        self.labels = labels
        self.size = len(data)
        self.dist = dist
        self.t = t
        self.centroids = pd.DataFrame(columns=data.columns.values)
        self.clusters = [[] for _ in range(k)]
        self.name = 'kmeans'
        self.has_converged = False

    def select_initial_centroids(self):
        random.seed(time.time_ns())

        for i in range(self.k):
            new_centroid_index = random.randint(0, self.size-1)

            for attr in self.data.columns.values:
                self.centroids.loc[i, attr] = self.data.loc[new_centroid_index, attr]

            self.clusters[i].append(new_centroid_index)

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

        if clusters == self.clusters:
            self.has_converged = True

        self.clusters = clusters

    def calc_centroids(self):
        for i, _ in enumerate(self.clusters):
            centroid = {}
            if len(self.clusters[i]) != 0:
                for attr in self.data.columns.values:
                    centroid[attr] = 0

                for point in self.clusters[i]:
                    for attr in self.data.columns.values:
                        centroid[attr] += self.data.loc[point][attr]

                for attr in self.data.columns.values:
                    centroid[attr] = centroid[attr]/len(self.clusters[i])

                self.centroids.loc[i] = centroid
            else:
                new_centroid_index = random.randint(0, self.size - 1)
                for attr in self.data.columns.values:
                    self.centroids.loc[i, attr] = self.data.loc[new_centroid_index, attr]


    def run(self):
        self.select_initial_centroids()

        i = 0
        while i < self.size and not self.has_converged:
            self.assign_clusters()
            self.calc_centroids()
            i += 1
