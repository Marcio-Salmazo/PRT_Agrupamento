import pandas as pd
import math


def euclidean_dist(x, y):
    return math.sqrt(math.pow(x["sepal_length"] - y["sepal_length"], 2) +
                     math.pow(x["sepal_width"] - y["sepal_width"], 2) +
                     math.pow(x["petal_length"] - y["petal_length"], 2) +
                     math.pow(x["petal_width"] - y["petal_width"], 2))


class AHC:

    def __init__(self, data, k=1):
        self.k = k
        self.size = len(data)
        self.data = data
        self.dist_matrix = self.calc_dist_matrix(data)
        self.clusters = self.init_clusters()
        self.output = open("output.txt", "w")

    def calc_dist_matrix(self, data):
        dist_matrix = [[0] * self.size for _ in range(self.size)]

        for i in range(self.size):
            for j in range(self.size):
                dist_matrix[i][j] = euclidean_dist(data.loc[i], data.loc[j])

        return dist_matrix

    def init_clusters(self):
        clusters = []

        for i in range(self.size):
            clusters.append([i])

        return clusters

    def find_single_link(self):
        min_dist = float('inf')
        clusters_to_merge = []

        for i in range(self.size):
            for j in range(i + 1, self.size):
                if self.dist_matrix[i][j] <= min_dist and i != j:
                    min_dist = self.dist_matrix[i][j]
                    clusters_to_merge = [i, j]

        return clusters_to_merge

    def merge_clusters(self, i, j):
        merge = self.clusters[i] + self.clusters[j]
        self.clusters[i] = merge
        self.clusters[j] = merge
        del self.clusters[j]

    def update_dist_matrix(self, i, j):
        i_dist = self.dist_matrix[i]
        j_dist = self.dist_matrix[j]
        merge_dist = []

        # O(n)
        for k in range(self.size):
            merge_dist_min = i_dist[k] if i_dist[k] < j_dist[k] else j_dist[k]
            merge_dist.append(merge_dist_min)

        self.dist_matrix[i] = merge_dist
        del self.dist_matrix[j]

        # O(n)
        for k in range(self.size - 1):
            k_dist = self.dist_matrix[k]
            new_min = k_dist[i] if k_dist[i] < merge_dist[k] else merge_dist[k]
            self.dist_matrix[k][i] = new_min
            del self.dist_matrix[k][j]

        self.size = len(self.dist_matrix)

    def print_clusters(self):
        for cluster in self.clusters:
            self.output.write(str(cluster).replace('[', '{').replace(']', '}') + ", ")
        self.output.write('\n')

    def run(self):
        while len(self.clusters) != self.k:
            self.print_clusters()
            clusters_to_merge = self.find_single_link()
            self.merge_clusters(clusters_to_merge[0], clusters_to_merge[1])
            self.update_dist_matrix(clusters_to_merge[0], clusters_to_merge[1])

        self.print_clusters()
        self.output.close()


def load_data(filename):
    return pd.read_csv(filename)


data_set = load_data("iris.csv")
data_set.drop("class", axis=1, inplace=True)
ahc = AHC(data_set)
ahc.run()
