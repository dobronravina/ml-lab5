import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from tqdm.auto import tqdm

data = pd.read_csv('WQ-R.csv', sep=';')

rows_count = data.shape[0]
columns_count = data.shape[1]
print(f'Кількість записів: {rows_count}')

data.drop(columns='quality', inplace=True)

for idx, column in enumerate(data.columns, 1):
    print(f'{idx}) {column}')


def plot_errors(errors, title, xlabel, ylabel):
    errors = np.array(errors)
    plt.figure(figsize=(13, 7))
    plt.plot(errors[:, 0], errors[:, 1], marker='o', linestyle='-', color='b', label='Error value')
    plt.xticks(errors[:, 0])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title(title)
    plt.legend()
    plt.style.use('ggplot')
    plt.show()


inertia_vals = []
silhouette_vals = []
centers = []

for clusters in range(1, 11):
    kmeans_model = KMeans(n_clusters=clusters, init='random', n_init=1, random_state=1).fit(data)
    inertia_vals.append([clusters, kmeans_model.inertia_])
    if clusters > 1:
        silhouette_vals.append([clusters, silhouette_score(data, kmeans_model.labels_)])
        centers.append(kmeans_model.cluster_centers_)

plot_errors(inertia_vals, 'Elbow Method', 'Number of Clusters', 'Inertia')
plot_errors(silhouette_vals, 'Average Silhouette Method', 'Number of Clusters', 'Silhouette Score')

splits = ShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
train_idx, test_idx = next(splits.split(data))
train_data, test_data = data.iloc[train_idx], data.iloc[test_idx]


def calculate_prediction_strength(k, train_centers, train_predictions, test_data, test_labels):
    test_size = len(test_data)
    D_matrix = np.zeros((test_size, test_size))
    for i in range(test_size):
        for j in range(i + 1, test_size):
            D_matrix[i, j] = D_matrix[j, i] = int(train_predictions[i] == train_predictions[j])

    strengths = []
    for cluster in range(k):
        intra_cluster_sum = 0
        cluster_size = np.sum(test_labels == cluster)
        if cluster_size > 1:
            for l1, idx1 in zip(test_labels, range(test_size)):
                for l2, idx2 in zip(test_labels, range(test_size)):
                    if l1 == l2 == cluster:
                        intra_cluster_sum += D_matrix[idx1, idx2]
            strengths.append(intra_cluster_sum / (cluster_size * (cluster_size - 1)))

    return min(strengths) if strengths else 0


strength_vals = []
for k in tqdm(range(1, 11)):
    kmeans_train = KMeans(n_clusters=k, init='random', n_init=1, random_state=1).fit(train_data)
    kmeans_test = KMeans(n_clusters=k, init='random', n_init=1, random_state=1).fit(test_data)
    pred_strength = calculate_prediction_strength(k, kmeans_train.cluster_centers_, kmeans_train.predict(test_data),
                                                  test_data, kmeans_test.labels_)
    strength_vals.append(pred_strength)

plt.figure(figsize=(13, 7))
plt.plot(range(1, 11), strength_vals, marker='o', linestyle='-', color='g', label='Prediction Strength')
plt.axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='Threshold (0.8)')
plt.title('Determining the optimal number of clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Prediction Strength')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.style.use('ggplot')
plt.show()

optimal_clusters = 2
best_kmeans_model = None
best_silhouette = -1
silhouette_vals = []
for i in range(10):
    kmeans_model = KMeans(n_clusters=optimal_clusters, init='k-means++', n_init=1).fit(data)
    silhouette_avg = silhouette_score(data, kmeans_model.labels_)
    silhouette_vals.append([i, silhouette_avg])
    if silhouette_avg > best_silhouette:
        best_kmeans_model = kmeans_model
        best_silhouette = silhouette_avg

print(f'Найкраща модель: {best_kmeans_model}')
print(f'Silhouette Score: {best_silhouette}')

plot_errors(silhouette_vals, 'Silhouette Score Detection', 'Iteration', 'Silhouette Score')

agglomerative_model = AgglomerativeClustering(n_clusters=optimal_clusters).fit(data)
cluster_centers_aggl = [data[agglomerative_model.labels_ == label].mean().values for label in
                        np.unique(agglomerative_model.labels_)]

print("Координати центрів кластерів AgglomerativeClustering:")
for center in cluster_centers_aggl:
    print(center)

agglo_silhouette = silhouette_score(data, agglomerative_model.labels_)
print(f'Silhouette Score AgglomerativeClustering: {agglo_silhouette}')
print(f'Silhouette Score KMeans: {best_silhouette}')
