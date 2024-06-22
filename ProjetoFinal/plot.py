import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('experiments_statistics.csv')
kmeans = df[df['algorithm'].isin(['kmeans'])]
ahc = df[df['algorithm'].isin(['ahc'])]
affinity = df[df['algorithm'].isin(['affinity'])]
data_sets = [s.split('.csv')[0] for s in df['data_set'].unique()]

fig, ax = plt.subplots()
ax.plot(data_sets, kmeans['silhouette_mean'], label='kmeans')
ax.plot(data_sets, ahc['silhouette_mean'], label='ahc')
ax.plot(data_sets, affinity['silhouette_mean'], label='affinity')
ax.set_title('Silhouette')
plt.ylim(0, 1)
plt.show()
plt.legend()

# plt.clf()
fig, ax = plt.subplots()
ax.plot(data_sets, kmeans['purity_mean'], label='kmeans')
ax.plot(data_sets, ahc['purity_mean'], label='ahc')
ax.plot(data_sets, affinity['purity_mean'], label='affinity')
ax.set_title('Purity')
plt.ylim(0, 1)
plt.show()
plt.legend()

fig, ax = plt.subplots()
ax.plot(data_sets, kmeans['duration_mean'], label='kmeans')
ax.plot(data_sets, ahc['duration_mean'], label='ahc')
ax.plot(data_sets, affinity['duration_mean'], label='affinity')
ax.set_title('Duration')
plt.show()
plt.legend()
