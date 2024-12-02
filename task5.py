# Практичне завдання: Аналіз клієнтської бази торгового центру методами кластеризації

# Імпортуємо необхідні бібліотеки
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, pairwise_distances

# Завантаження датасету
df = pd.read_csv('Mall_Customers.csv')

# Частина 1: Підготовка даних

# Перевірка на пропущені значення
print("Перевірка на пропущені значення:")
print(df.isnull().sum())

# Описова статистика
print("\nОписова статистика:")
print(df.describe())

# Побудова гістограм для кожної змінної
df.hist(bins=20, figsize=(12, 10))
plt.show()

# Стандартизація даних (стандартизуємо лише числові змінні)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Age', 'Annual_Income', 'Spending_Score']])

# Створення нового датафрейму з стандартизованими даними
df_scaled = pd.DataFrame(df_scaled, columns=['Age', 'Annual_Income', 'Spending_Score'])

# Частина 2: Визначення оптимальної кількості кластерів

# Лікоть (Elbow method)
inertia = []
sil_scores = []

# Тестуємо кількість кластерів від 1 до 10
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)
    if k > 1:
        sil_scores.append(silhouette_score(df_scaled, kmeans.labels_))

# Графік інерції
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Метод ліктя (Elbow method)')
plt.xlabel('Кількість кластерів')
plt.ylabel('Інерція')
plt.show()

# Графік коефіцієнта силуету
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), sil_scores, marker='o')
plt.title('Коефіцієнт силуету')
plt.xlabel('Кількість кластерів')
plt.ylabel('Silhouette Score')
plt.show()

# Визначення оптимальної кількості кластерів
optimal_k = 5  # Припустимо, що оптимальна кількість кластерів це 5 (на основі ліктя та силуету)

# Частина 3: Кластеризація та аналіз результатів

# Кластеризація методом K-means
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Візуалізація результатів
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Annual_Income', y='Spending_Score', hue='Cluster', palette='Set2', s=100)
plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], s=200, c='black', marker='X')
plt.title('Кластеризація методом K-means')
plt.show()

# Аналіз середніх значень показників для кожного кластера
cluster_means = df.groupby('Cluster')[['Age', 'Annual_Income', 'Spending_Score']].mean()
print("\nСередні значення для кожного кластера:")
print(cluster_means)

# Опис характеристик кожного кластера
for i in range(optimal_k):
    print(f"\nКластер {i}:")
    cluster_data = df[df['Cluster'] == i]
    print(f"Середній вік: {cluster_data['Age'].mean()}")
    print(f"Середній дохід: {cluster_data['Annual_Income'].mean()}")
    print(f"Середній бал лояльності: {cluster_data['Spending_Score'].mean()}")

# Частина 4: Додаткові завдання

# Порівняння результатів K-means з іншими методами кластеризації (DBSCAN, ієрархічна кластеризація)

# DBSCAN кластеризація
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(df_scaled)

# Ієрархічна кластеризація
agg_clustering = AgglomerativeClustering(n_clusters=optimal_k)
df['Agglomerative_Cluster'] = agg_clustering.fit_predict(df_scaled)

# Візуалізація результатів кластеризацій
plt.figure(figsize=(15, 6))

# K-means
plt.subplot(1, 3, 1)
sns.scatterplot(data=df, x='Annual_Income', y='Spending_Score', hue='Cluster', palette='Set2', s=100)
plt.title('K-means Clustering')

# DBSCAN
plt.subplot(1, 3, 2)
sns.scatterplot(data=df, x='Annual_Income', y='Spending_Score', hue='DBSCAN_Cluster', palette='Set2', s=100)
plt.title('DBSCAN Clustering')

# Ієрархічна кластеризація
plt.subplot(1, 3, 3)
sns.scatterplot(data=df, x='Annual_Income', y='Spending_Score', hue='Agglomerative_Cluster', palette='Set2', s=100)
plt.title('Agglomerative Clustering')

plt.tight_layout()
plt.show()

# Додаткові метрики для оцінки якості кластеризації

# Розрахунок Dunn Index
def dunn_index(X, labels):
    distances = pairwise_distances(X)
    intra_cluster_distances = []
    inter_cluster_distances = []
    
    for cluster in np.unique(labels):
        cluster_points = X[labels == cluster]
        intra_cluster_distances.append(np.max(pairwise_distances(cluster_points)))
        
        for other_cluster in np.unique(labels[labels != cluster]):
            other_cluster_points = X[labels == other_cluster]
            inter_cluster_distances.append(np.min(pairwise_distances(cluster_points, other_cluster_points)))
    
    return np.min(inter_cluster_distances) / np.max(intra_cluster_distances)

dunn_score = dunn_index(df_scaled, df['Cluster'])
print(f"\nDunn Index: {dunn_score}")
