def silhouette(clusters, df, dist):
    point_sil_per_cluster = {}

    for cluster in range(len(clusters)):
        point_sil_per_cluster[cluster] = {}
        for i in range(len(clusters[cluster])):
            point = clusters[cluster][i]

            other_points = clusters[cluster].copy()
            del other_points[i]

            other_clusters = clusters.copy()
            del other_clusters[cluster]

            point_a = 0
            point_bs = []

            for other_point in other_points:
                point_a += dist(df.loc[point], df.loc[other_point], df.columns.values)
            point_a = point_a/len(clusters[cluster])

            for other_cluster in other_clusters:
                point_b = 0
                for other_point in other_cluster:
                    point_b += dist(df.loc[point], df.loc[other_point], df.columns.values)
                point_b = point_b/len(other_cluster)
                point_bs.append(point_b)

            point_b = min(point_bs)
            point_s = (point_b-point_a)/max(point_a, point_b)
            point_sil_per_cluster[cluster][point]=point_s

    clusters_sil = {}
    clustering_sil = 0
    for cluster in point_sil_per_cluster:
        clusters_sil[cluster] = 0

        for p in point_sil_per_cluster[cluster]:
            clusters_sil[cluster] += point_sil_per_cluster[cluster][p]
            clustering_sil += point_sil_per_cluster[cluster][p]

        clusters_sil[cluster] = clusters_sil[cluster]/len(point_sil_per_cluster[cluster])

    clusters_sil["clustering silhouette"] = clustering_sil/len(df)
    return clusters_sil


def purity(clusters, labels, classes):
    clusters_purities = {}

    for cluster in range(len(clusters)):
        clusters_purities[cluster] = {}

        for label in classes:
            clusters_purities[cluster][label] = 0

        for point in clusters[cluster]:
            clusters_purities[cluster][labels.loc[point]] += 1

        for label in classes:
            clusters_purities[cluster][label] = clusters_purities[cluster][label]/len(clusters[cluster])

    result = {}
    for cluster, purities in clusters_purities.items():
        result[cluster] = max(purities.values())

    clustering_p = 0
    for c, p in result.items():
        clustering_p += (len(clusters[c])/len(labels))*p

    result["clustering purity"] = clustering_p
    return result
