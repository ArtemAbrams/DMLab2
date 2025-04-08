import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 1. Loading Data: Завантаження даних із CSV файлу
df = pd.read_csv('customers.csv')
print("Перші 5 рядків початкового набору даних:")
print(df.head())

print("\nІнформація про дані:")
print(df.info())

print("\nКількість пропущених значень по стовпцях:")
print(df.isnull().sum())

# Видаляємо пропущені значення
df = df.dropna()

# 2. Data Cleaning: Видалення аномалій (outliers) для числових змінних за допомогою IQR
numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

print("\nОпис статистики даних після видалення аномалій:")
print(df.describe())

# 3. Feature Selection: Вибір ознак для кластеризації (без Gender)
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]
print("\nВибрані ознаки для кластеризації:")
print(X.head())

# 4. Data Normalization: Масштабування даних за допомогою StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nНормалізовані дані (перші 5 зразків):")
print(X_scaled[:5])

# ============================
# Exploratory Data Analysis (EDA)
# ============================

# Побудова гістограм для аналізу розподілу ключових змінних
plt.figure(figsize=(14, 4))

plt.subplot(1, 3, 1)
plt.hist(df['Age'], bins=10, edgecolor='black')
plt.title('Розподіл віку')
plt.xlabel('Вік')
plt.ylabel('Кількість')

plt.subplot(1, 3, 2)
plt.hist(df['Annual Income (k$)'], bins=10, edgecolor='black')
plt.title('Розподіл річного доходу')
plt.xlabel('Річний дохід (k$)')

plt.subplot(1, 3, 3)
plt.hist(df['Spending Score (1-100)'], bins=10, edgecolor='black')
plt.title('Розподіл Spending Score')
plt.xlabel('Spending Score (1-100)')

plt.tight_layout()
plt.show()

# Побудова scatter plot для аналізу взаємозв’язків між ознаками
plt.figure(figsize=(14, 4))

plt.subplot(1, 3, 1)
plt.scatter(df['Age'], df['Annual Income (k$)'], alpha=0.6)
plt.title('Вік vs Річний дохід')
plt.xlabel('Вік')
plt.ylabel('Річний дохід (k$)')

plt.subplot(1, 3, 2)
plt.scatter(df['Age'], df['Spending Score (1-100)'], alpha=0.6)
plt.title('Вік vs Spending Score')
plt.xlabel('Вік')
plt.ylabel('Spending Score (1-100)')

plt.subplot(1, 3, 3)
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], alpha=0.6)
plt.title('Річний дохід vs Spending Score')
plt.xlabel('Річний дохід (k$)')
plt.ylabel('Spending Score (1-100)')

plt.tight_layout()
plt.show()

# ============================
# 3. Implementing K-means Clustering
# ============================

# 3.1. Вибір кількості кластерів (k)

# Метод ліктя: обчислення WCSS для k від 1 до 10
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Кількість кластерів (k)')
plt.ylabel('WCSS')
plt.title('Метод ліктя')
plt.show()

# Silhouette Analysis для k від 2 до 10
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(score)

plt.figure(figsize=(6, 4))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Кількість кластерів (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.show()

# На основі графіків обираємо оптимальне значення k (наприклад, k = 5)
optimal_k = 5

# 3.2. Кластеризація за допомогою K-means
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

print("Кількість елементів у кожному кластері:")
print(df['Cluster'].value_counts())

# 3.3. Аналіз кластерів: обчислюємо середні значення ознак для кожного кластеру
cluster_analysis = df.groupby('Cluster')[features].mean()
print("\nСередні значення ознак для кожного кластеру:")
print(cluster_analysis)

# ============================
# 4. Evaluation of Clusters
# ============================

# Розрахунок Silhouette Score для отриманих кластерів
score = silhouette_score(X_scaled, clusters)
print("\nSilhouette Score:", score)

# Аналіз кластерів: обчислення середніх значень ознак для кожного кластеру
cluster_analysis = df.groupby('Cluster')[features].mean()
print("\nСередні значення ознак для кожного кластеру:")
print(cluster_analysis)

print("\nКількість елементів у кожному кластері:")
print(df['Cluster'].value_counts())

# ============================
# 5. Visualization of Clusters using PCA
# ============================

# Виконуємо PCA для зменшення розмірності до 2 компонент
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Створюємо scatter plot кластерів в просторі PCA
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.xlabel("Перший головний компонент")
plt.ylabel("Другий головний компонент")
plt.title("Візуалізація кластерів після PCA")
plt.legend(*scatter.legend_elements(), title="Кластери")
plt.show()

# ============================
# 6. Visual Representation of Cluster Characteristics
# ============================

# Обчислюємо середні значення ознак для кожного кластеру
cluster_means = df.groupby('Cluster')[features].mean()

# Побудова bar chart для візуалізації середніх значень ознак по кластерах
cluster_means.plot(kind='bar', figsize=(10, 6))
plt.title("Середні значення ознак для кожного кластеру")
plt.xlabel("Ім'я кластеру")
plt.ylabel("Середнє значення")
plt.legend(title="Ознаки")
plt.show()
