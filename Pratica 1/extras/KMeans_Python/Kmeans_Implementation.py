from sklearn.preprocessing import MinMaxScaler
from tkinter import filedialog
import tkinter as tk
import pandas as pd
import numpy as np
import math


class K_Means:

    def __init__(self, k=2, tolerance=0.001, max_iter=100):
        self.k = k
        self.max_iterations = max_iter
        self.tolerance = tolerance

    def euclidean_distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def predict(self, point):
        """
            É realizado o calculo das distâncias entre um determinado ponto
            e cada um dos centróides definidos pelo agrupamento. O cluster que
            resultar na menor distância sera o resultado da predição

            obs: A função np.linalg.norm() retorna a distância euclidiana,
                 o uso desta função é mais eficiente para este calculo
        """
        distances = [np.linalg.norm(point - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

    def fit(self, data):

        """
            A inicialização dos centróides é feita de maneira simples:
            * São escolhidos K centróides aleatoriamente do dataset.
        """
        self.centroids = {}  # Inicialização do dicionário de centroides

        np.random.seed(17)
        for i in range(self.k):
            self.centroids[i] = data[np.random.choice(len(data), 1, replace=False)[0]]

        for i in range(self.max_iterations):

            self.classes = {}  # Inicialização do dicionário de classes

            for j in range(self.k):
                self.classes[j] = []  # Inicialização do array de distâncias

            '''  
                Sera computada as distancias euclidianas de um determinado 
                ponto i com todos os centróides definidos. Após o preenchimento
                da lista de distâncias, será definido o cluster_index, que 
                representa o índice do cluster que obteve a menor distância 
                euclidiana com o ponto (O mesmo cluster_index também será 
                utilizado como index da 'classe' para definir para qual cluster
                o ponto será alocado).  

                Obs: O mesmo processo será repetido para todos os pontos do dataset

            '''
            for point in data:
                distances = []
                for index in self.centroids:
                    distances.append(self.euclidean_distance(point, self.centroids[index]))
                cluster_index = distances.index(min(distances))
                self.classes[cluster_index].append(point)

            '''  
                Previous é um dicionário python com base nos centróides, que 
                armazena os valores prévios dos centróides definidos
            '''
            previous = self.centroids

            ''' 
                Para cada índice de cluster dentro do dicionário de classes
                é calculado um novo valor de centróide, levando em consideraço a média 
                dos valores dos pontos que foram alocados do dicionário 'classes' no 
                trecho de código anterior: self.classes[cluster_index].append(point)

                obs: A flag isOptimal é definida como True
            '''
            for cluster_index in self.classes:
                self.centroids[cluster_index] = np.average(self.classes[cluster_index], axis=0)
            isOptimal = True

            '''  
                Para cada centroid definido, será avaliada o qual foi a variação
                de valor do novo centróide calculado. Se a porcentagem da varição
                for superior à tolerância estabelecida previamente, o loop de execução
                continua, caso contrário (isOptimal is True), a execução é encerrada. 
            '''
            for centroid in self.centroids:
                original_centroid = previous[centroid]
                curr = self.centroids[centroid]
                if np.sum((curr - original_centroid) / original_centroid * 100.0) > self.tolerance:
                    isOptimal = False
            if isOptimal:
                break


def select_file():
    root = tk.Tk()
    root.withdraw()
    arquivo = filedialog.askopenfilename(title="Selecione um arquivo .csv",
                                         filetypes=(("Arquivos CSV", "*.csv"),
                                                    ("Todos os arquivos", "*.*")))
    return arquivo


def dropClass(column, dataset):
    dataset.drop(column, axis=1, inplace=True)
    return dataset


def normalize(dataset):
    return (dataset - dataset.min()) / (dataset.max() - dataset.min())


# -----------------------------------------------------------------------------------------

dataframe = select_file()
datasetPD = pd.read_csv(dataframe)

classColumn = input("Nome da coluna de classes a ser excluída")
if classColumn in datasetPD:
    datasetPD = dropClass(classColumn, datasetPD)
else:
    datasetPD = datasetPD
print(datasetPD)

# -----------------------------------------------------------------------------------------

count = 0
for i in range(0, datasetPD.shape[1]):
    if datasetPD[datasetPD.columns[i]].dtype == 'float64' or datasetPD[datasetPD.columns[i]].dtype == 'int64':
        count += 1

datasetNorm = datasetPD.copy()

if count == datasetPD.shape[1]:
    scaler = MinMaxScaler()
    datasetNorm = scaler.fit_transform(datasetPD)

    # Exibindo os 5 primeiros elementos do dataset
    # a fim de verificar a aplicação da normalização
    print(datasetNorm[:5], type(datasetNorm))

else:
    print('A base de dados contém atributos não numerico, não foi possível normalizar')

# -----------------------------------------------------------------------------------------

'''  
    Definição da instância da classe k_means criada
    e treinamento com a base de dados iris_dataset
'''
k_means = K_Means(k=3, tolerance=0.001, max_iter=100)
k_means.fit(datasetNorm)

# -----------------------------------------------------------------------------------------

''' 
    Os dados de predição serão adicionados 
    ao dataset original (como uma coluna extra
    indicado o cluster ao qual o objeto pertence)
    e o novo conteúdo será exportado em formato CSV.
'''
classes = []
for i in range(len(datasetNorm)):
    classes.append(k_means.predict(datasetNorm[i]))

OutputCSV = pd.read_csv(dataframe)

if 'Grupo' not in OutputCSV:
    OutputCSV.insert(OutputCSV.shape[1], 'Grupo', classes, False)

else:
    print('Classificação já atribuída')

OutputCSV.to_csv('Resultado.csv')
print(OutputCSV)
