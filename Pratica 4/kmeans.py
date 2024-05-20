import math
import numpy as np
from typing import Tuple


class K_Means:
    def __init__(self, k=2, tolerance=0.001, max_iter=100):
        self.k = k
        self.max_iterations = max_iter
        self.tolerance = tolerance

    def euclidean_distance(self, point1, point2) -> float:
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def predict(self, point) -> int:
        """
        É realizado o calculo das distâncias entre um determinado ponto
        e cada um dos centróides definidos pelo agrupamento. O cluster que
        resultar na menor distância sera o resultado da predição

        obs: A função np.linalg.norm() retorna a distância euclidiana,
             o uso desta função é mais eficiente para este calculo
        """
        distances = [
            np.linalg.norm(point - centroid) for centroid in self.centroids.values()
        ]
        classification = np.argmin(distances)
        return classification

    def fit(self, data: np.ndarray):
        """
        A inicialização dos centróides é feita de maneira simples:
        * São escolhidos K centróides aleatoriamente do dataset.
        """
        self.centroids = {}  # Inicialização do dicionário de centroides
        for i in range(self.k):
            idx = np.random.choice(len(data), 1, replace=False)[0]
            self.centroids[i] = data[idx]

        for i in range(self.max_iterations):
            # Inicialização do dicionário de classes
            self.classes = {j: [] for j in range(self.k)}

            """  
                Sera computada as distancias euclidianas de um determinado 
                ponto i com todos os centróides definidos. Após o preenchimento
                da lista de distâncias, será definido o cluster_index, que 
                representa o índice do cluster que obteve a menor distância 
                euclidiana com o ponto (O mesmo cluster_index também será 
                utilizado como index da 'classe' para definir para qual cluster
                o ponto será alocado).  

                Obs: O mesmo processo será repetido para todos os pontos do dataset

            """
            for point in data:
                distances = []
                for index in self.centroids:
                    distances.append(
                        self.euclidean_distance(point, self.centroids[index])
                    )
                cluster_index = distances.index(min(distances))
                self.classes[cluster_index].append(point)

            """  
                Previous é um dicionário python com base nos centróides, que 
                armazena os valores prévios dos centróides definidos
            """
            previous = self.centroids

            """ 
                Para cada índice de cluster dentro do dicionário de classes
                é calculado um novo valor de centróide, levando em consideraço a média 
                dos valores dos pontos que foram alocados do dicionário 'classes' no 
                trecho de código anterior: self.classes[cluster_index].append(point)

                obs: A flag is_optimal é definida como True
            """
            for cluster_index in self.classes:
                self.centroids[cluster_index] = np.average(
                    self.classes[cluster_index], axis=0
                )

            is_optimal = True

            """  
                Para cada centroid definido, será avaliada o qual foi a variação
                de valor do novo centróide calculado. Se a porcentagem da varição
                for superior à tolerância estabelecida previamente, o loop de execução
                continua, caso contrário (is_optimal is True), a execução é encerrada. 
            """
            for centroid in self.centroids:
                original_centroid = previous[centroid]
                curr = self.centroids[centroid]
                if (
                    np.sum((curr - original_centroid) / original_centroid * 100.0)
                    > self.tolerance
                ):
                    is_optimal = False

            if is_optimal:
                break

    def print_centroids(self):
        for key, value in self.centroids.items():
            print(f"Centroid {key}: {value}")

    def fit_predict(self, data) -> Tuple[np.array, np.ndarray]:
        self.fit(data)
        labels = np.array([self.predict(point) for point in data])
        return labels, self.centroid_values

    @property
    def centroid_values(self) -> np.array:
        # return np.array(list(self.centroids.values()))
        return list(self.centroids.values())


def kmeans_v2(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    # random initialize centroids
    sampled_idxs = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[sampled_idxs]
    new_centroids = np.zeros((k, X.shape[1]))
    labels = np.zeros(X.shape[0])

    # TODO: max iterations?
    while not np.all(centroids == new_centroids):
        # update cluster membership
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)  # closest

        # update centroids
        new_centroids = np.array([np.mean(X[labels == i], axis=0) for i in range(k)])

        centroids = new_centroids.copy()

    return labels, centroids
