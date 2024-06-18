import math


# Distância euclideana entre dois pontos quaisquer, é necessário passar a lista de atributos dos pontos
def euclidean_dist(x, y, attrs):
    acc = 0
    for attr in attrs:
        acc += math.pow(x[attr] - y[attr], 2)

    return math.sqrt(acc)


def transform_to_dissimilarity(similarity):
    return 1-similarity


def cosine_similarity(x, y, attrs):
    xy = 0
    xlength = 0
    ylength = 0

    for attr in attrs:
        xy += x[attr] * y[attr]
        xlength += math.pow(x[attr], 2)
        ylength += math.pow(y[attr], 2)

    return transform_to_dissimilarity(xy/(math.sqrt(xlength)*math.sqrt(ylength)))
