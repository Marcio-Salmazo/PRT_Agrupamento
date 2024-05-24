import numpy as np
import pandas as pd

from ahc import AHC
from kmeans import K_Means


def simplified_silhouette(X, labels):
    n = X.shape[0]
    sil_scores = np.zeros(n)

    for i in range(n):
        # encontra o cluster ao qual i pertence
        own_cluster = labels[i]

        # a_i: distancia media intra-cluster,
        # media das distancias de i para todos os outros objs no mesmo cluster
        a_i = np.mean(np.linalg.norm(X[i] - X[labels == own_cluster], axis=1))

        # b_i: distancia media mais proxima inter-cluster,
        b_i = np.inf
        for label in np.unique(labels):
            if label == own_cluster:
                continue
            # para cada cluster diferente de i
            # 1. calcula a media das distancias de i para todos objetos no cluster
            # 2. calcula o minimo dessas distancias
            b_i = min(b_i, np.mean(np.linalg.norm(X[i] - X[labels == label], axis=1)))

        # calcula o valor da silhueta simplificada para i,
        # se o tamanho do cluster de i for maior que 1, caso contrario 0
        sil_scores[i] = (
            (b_i - a_i) / max(a_i, b_i) if np.sum(labels == own_cluster) > 1 else 0
        )

    return np.mean(sil_scores)


if __name__ == "__main__":
    # Carregando dataset
    filename = "./iris.csv"
    iris = pd.read_csv(filename)

    # Ignorando a coluna `class`
    X = iris.iloc[:, :4]

    # Numero de clusters
    k = 3
    # k = int(input("Numero de clusters: "))

    print("k\t\t\t", k)
    print("Dataset\t\t\t", filename)
    print("Dataset shape\t\t", X.shape)

    model = K_Means(k=k)
    kmeans_labels, kmeans_centroids = model.fit_predict(X.to_numpy())
    kmeans_sil = simplified_silhouette(X.to_numpy(), kmeans_labels)

    print("\n=== K-Means ===")
    print("Silhouette\t ", kmeans_sil)

    ahc = AHC(data=X, k=k)
    ahc.run()
    ahc_sil = simplified_silhouette(X.to_numpy(), ahc.labels)
    print("\n=== AHC ===")
    print("Silhouette\t ", ahc_sil)
