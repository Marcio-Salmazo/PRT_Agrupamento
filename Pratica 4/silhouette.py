import numpy as np
import pandas as pd

from ahc import AHC
from kmeans import K_Means


def silhouette_score(X: np.ndarray, labels: np.array):
    # number of samples
    n = X.shape[0]

    # initialize silhouette scores
    sil_scores = np.zeros(n)

    # calculate pairwise distances manually
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.sqrt(np.sum((X[i] - X[j]) ** 2))

    for i in range(n):
        # find the cluster of the current sample
        own_cluster = labels[i]

        # mask for the own cluster
        mask_own = labels == own_cluster

        # a(i): the average distance to all other points in the same cluster
        if np.sum(mask_own) > 1:
            a_i = np.sum(D[i, mask_own]) / (np.sum(mask_own) - 1)
        else:
            a_i = 0

        # b(i): the minimum average distance to points in any other cluster
        b_i = np.inf
        for label in np.unique(labels):
            if label == own_cluster:
                continue
            mask_other = labels == label
            b_i = min(b_i, np.mean(D[i, mask_other]))

        # compute the silhouette score for the current sample
        sil_scores[i] = (b_i - a_i) / max(a_i, b_i)

    return np.mean(sil_scores)


if __name__ == "__main__":
    filename = "iris.csv"
    iris = pd.read_csv(filename)

    X = iris.iloc[:, :4]
    # targets
    # y = iris.iloc[:, 4]

    k = 3
    # k = int(input("Number of clusters: "))
    print("dataset: ", filename)
    print("k: ", k)

    print("\n===")
    print("K-Means")
    model = K_Means(k=k)
    labels, centroids = model.fit_predict(X.to_numpy())
    model.print_centroids()
    print("Labels: ")
    print(labels)
    print("Silhouette Score: ", silhouette_score(X.to_numpy(), labels))
    # from sklearn.metrics import silhouette_score
    # print("sklearn Silhouette Score:", silhouette_score(X.to_numpy(), labels))

    # TODO:
    print("\n===")
    print("AHC")
    ahc = AHC(X, k)
    ahc.run()
