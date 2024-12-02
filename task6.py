# Імпортуємо необхідні бібліотеки
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Завантаження датасету
df = pd.read_csv('Mall_Customers.csv')

# Частина 1: Підготовка даних

# Перевірка на пропущені значення
print("Перевірка на пропущені значення:")
print(df.isnull().sum())

# Описова статистика
print("\nОписова статистика:")
print(df.describe())

# Кодування категоріальних змінних (наприклад, стать)
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Стандартизація даних (стандартизуємо лише числові змінні)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Age', 'Annual_Income', 'Spending_Score']])

# Створення нового датафрейму з стандартизованими даними
df_scaled = pd.DataFrame(df_scaled, columns=['Age', 'Annual_Income', 'Spending_Score'])

# Частина 2: Застосування PCA (Principal Component Analysis)

# Застосування PCA
pca = PCA()
pca.fit(df_scaled)

# Визначення оптимальної кількості головних компонент
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Графік поясненої дисперсії
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.title('Пояснена дисперсія головними компонентами')
plt.xlabel('Кількість компонент')
plt.ylabel('Кумулятивна пояснена дисперсія')
plt.grid(True)
plt.show()

# Вибір оптимальної кількості компонент (вибираємо 2 компоненти)
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Візуалізація результатів PCA в 2D просторі
plt.figure(figsize=(8, 6))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c='blue', edgecolor='k', s=50)
plt.title('Результати PCA (2 компоненти)')
plt.xlabel('Головна компонента 1')
plt.ylabel('Головна компонента 2')
plt.grid(True)
plt.show()

# Візуалізація результатів PCA в 3D просторі
from mpl_toolkits.mplot3d import Axes3D

pca = PCA(n_components=3)
df_pca_3d = pca.fit_transform(df_scaled)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_pca_3d[:, 0], df_pca_3d[:, 1], df_pca_3d[:, 2], c='blue', edgecolor='k', s=50)
ax.set_title('Результати PCA (3 компоненти)')
ax.set_xlabel('Головна компонента 1')
ax.set_ylabel('Головна компонента 2')
ax.set_zlabel('Головна компонента 3')
plt.show()

# Частина 3: Застосування t-SNE (t-Distributed Stochastic Neighbor Embedding)

# Застосування t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
df_tsne = tsne.fit_transform(df_scaled)

# Візуалізація результатів t-SNE
plt.figure(figsize=(8, 6))
plt.scatter(df_tsne[:, 0], df_tsne[:, 1], c='green', edgecolor='k', s=50)
plt.title('Результати t-SNE')
plt.xlabel('t-SNE компонент 1')
plt.ylabel('t-SNE компонент 2')
plt.grid(True)
plt.show()

# Частина 4: Кластеризація на зменшених даних

# Кластеризація K-means на результатах PCA
kmeans_pca = KMeans(n_clusters=5, random_state=42)
df['Cluster_PCA'] = kmeans_pca.fit_predict(df_pca)

# Кластеризація K-means на результатах t-SNE
kmeans_tsne = KMeans(n_clusters=5, random_state=42)
df['Cluster_tSNE'] = kmeans_tsne.fit_predict(df_tsne)

# Візуалізація результатів кластеризації на PCA
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_pca[:, 0], y=df_pca[:, 1], hue=df['Cluster_PCA'], palette='Set2', s=100)
plt.title('Кластеризація K-means на PCA')
plt.xlabel('Головна компонента 1')
plt.ylabel('Головна компонента 2')
plt.grid(True)
plt.show()

# Візуалізація результатів кластеризації на t-SNE
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_tsne[:, 0], y=df_tsne[:, 1], hue=df['Cluster_tSNE'], palette='Set2', s=100)
plt.title('Кластеризація K-means на t-SNE')
plt.xlabel('t-SNE компонент 1')
plt.ylabel('t-SNE компонент 2')
plt.grid(True)
plt.show()

# Частина 5: Порівняння методів

# Порівняння результатів кластеризації
print("\nSilhouette Score для PCA кластеризації:")
print(silhouette_score(df_pca, df['Cluster_PCA']))

print("\nSilhouette Score для t-SNE кластеризації:")
print(silhouette_score(df_tsne, df['Cluster_tSNE']))

# Частина 6: Інтерпретація результатів

# Аналіз груп клієнтів
for cluster in range(5):
    print(f"\nКластер {cluster} (PCA):")
    cluster_data = df[df['Cluster_PCA'] == cluster]
    print(f"Середній вік: {cluster_data['Age'].mean()}")
    print(f"Середній дохід: {cluster_data['Annual_Income'].mean()}")
    print(f"Середній бал лояльності: {cluster_data['Spending_Score'].mean()}")

    print(f"\nКластер {cluster} (t-SNE):")
    cluster_data = df[df['Cluster_tSNE'] == cluster]
    print(f"Середній вік: {cluster_data['Age'].mean()}")
    print(f"Середній дохід: {cluster_data['Annual_Income'].mean()}")
    print(f"Середній бал лояльності: {cluster_data['Spending_Score'].mean()}")

# Запропоновані маркетингові стратегії:
# Класифікація клієнтів на основі їх віку, доходу та лояльності може допомогти створити цільові рекламні кампанії:
# - Кластер 1: молоді клієнти з низьким доходом — спеціальні пропозиції для студентів або молоді.
# - Кластер 2: старші клієнти з високим доходом — преміум-продукти або клуби лояльності.
# - Кластер 3: середні за віком з високою лояльністю — спеціальні акції для постійних клієнтів.
