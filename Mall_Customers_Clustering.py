

# --- 1. SETUP AND DATA LOADING ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Try loading the dataset
try:
    df = pd.read_csv('Mall_Customers.csv')
    print(" Dataset loaded successfully!")
    print(df.head())
except FileNotFoundError:
    print(" Error: 'Mall_Customers.csv' not found.")
    print("Please ensure the file is uploaded or located in the same directory.")
    # The script continues for demonstration, assuming data is loaded

# --- 2. DATA CLEANING AND EXPLORATORY DATA ANALYSIS (EDA) ---

# Basic data overview
print("\n--- DATA INFORMATION ---")
df.info()

print("\n--- MISSING VALUES CHECK ---")
print(df.isnull().sum())

print("\n--- DUPLICATE RECORDS CHECK ---")
print(f"Number of duplicates: {df.duplicated().sum()}")

# Rename columns for convenience
df.rename(columns={
    'Annual Income (k$)': 'Annual_Income',
    'Spending Score (1-100)': 'Spending_Score'
}, inplace=True)

print("\n--- CLEANED DATA PREVIEW ---")
print(df.head())

# --- EXPLORATORY DATA ANALYSIS (EDA) ---
sns.set(style="whitegrid")

# 1. Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution of Customers')
plt.xlabel('Age')
plt.ylabel('Number of Customers')
plt.show()

# 2. Gender Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='Gender', data=df, palette='pastel')
plt.title('Gender Distribution')
plt.show()

# 3. Income and Spending Distributions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df['Annual_Income'], bins=30, kde=True, color='green')
plt.title('Annual Income Distribution')
plt.xlabel('Annual Income (k$)')

plt.subplot(1, 2, 2)
sns.histplot(df['Spending_Score'], bins=30, kde=True, color='purple')
plt.title('Spending Score Distribution')
plt.xlabel('Spending Score (1–100)')
plt.tight_layout()
plt.show()

# 4. Relationship between Income and Spending Score
plt.figure(figsize=(10, 7))
sns.scatterplot(
    x='Annual_Income',
    y='Spending_Score',
    data=df,
    hue='Gender',
    s=100,
    alpha=0.7
)
plt.title('Relationship between Annual Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1–100)')
plt.legend()
plt.show()

# --- 3. K-MEANS CLUSTERING ---


# 3a. CLUSTERING WITH ONE FEATURE: Annual Income

print("\n--- 3a. Clustering on 'Annual_Income' ---")

# Select and scale the feature
X1 = df[['Annual_Income']]
scaler1 = StandardScaler()
X1_scaled = scaler1.fit_transform(X1)

# Determine optimal k using Elbow Method
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X1_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method – 1D Clustering (Annual Income)')
plt.show()

# Evaluate clustering using Silhouette Scores
silhouette_scores = []
for k in range(2, 11):  # silhouette score needs at least 2 clusters
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X1_scaled)
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(X1_scaled, labels))

plt.figure(figsize=(10, 5))
plt.plot(range(2, 11), silhouette_scores, 'ro-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores – 1D Clustering (Annual Income)')
plt.show()

# Based on visual analysis, choose k = 3
kmeans1_final = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
df['Cluster_1D'] = kmeans1_final.fit_predict(X1_scaled)
print("✅ 1D Clustering complete (k=3).")



#  CLUSTERING WITH TWO FEATURES: Annual Income + Spending Score

print("\n--- 3b. Clustering on 'Annual_Income' and 'Spending_Score' ---")

# Select and scale the features
X2 = df[['Annual_Income', 'Spending_Score']]
scaler2 = StandardScaler()
X2_scaled = scaler2.fit_transform(X2)

# Elbow Method for 2D clustering
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X2_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method – 2D Clustering (Income + Spending)')
plt.show()

# Silhouette Score validation
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X2_scaled)
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(X2_scaled, labels))

plt.figure(figsize=(10, 5))
plt.plot(range(2, 11), silhouette_scores, 'ro-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores – 2D Clustering (Income + Spending)')
plt.show()

# Based on results, choose k = 5
kmeans2_final = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=42)
df['Cluster_2D'] = kmeans2_final.fit_predict(X2_scaled)
print(" 2D Clustering complete (k=5).")

# Display updated dataset
print("\n--- DATA WITH CLUSTER LABELS ---")
print(df.head())

# --- 4. VISUALIZATION AND COMPARISON OF CLUSTERS ---
print("\n--- 4. Visualizing and Comparing Clusters ---")

plt.figure(figsize=(18, 8))

# 1 Clustering Results
plt.subplot(1, 2, 1)
sns.stripplot(x='Cluster_1D', y='Annual_Income', data=df, palette='viridis', jitter=True)
plt.title('1D Clustering (k=3) – Based on Annual Income')
plt.xlabel('Cluster')
plt.ylabel('Annual Income (k$)')

# 2 Clustering Results
plt.subplot(1, 2, 2)
sns.scatterplot(
    x='Annual_Income',
    y='Spending_Score',
    hue='Cluster_2D',
    data=df,
    palette='bright',
    s=100,
    alpha=0.9
)
plt.title('2D Clustering (k=5) – Based on Income and Spending')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1–100)')
plt.legend(title='Cluster')

plt.tight_layout()
plt.show()

print("\n ANALYSIS COMPLETE.")
