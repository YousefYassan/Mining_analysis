import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import plotly.express as px
import matplotlib.pyplot as plt
# Load the dataset
data = pd.read_csv("Mall_Customers.csv")

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())
# # Data Preprocessing:
# Step 1: Handling missing values (if any)
missing_values = data.isnull().sum()
print("Missing values:\n", missing_values)
# Step 2: Handling outliers (if any)
# Select numerical columns for box plot
numerical_columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

# Create box plots for each numerical column
plt.figure(figsize=(10, 6))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(y=data[column], color='skyblue')
    plt.title(column)
plt.tight_layout()
plt.show()
# Step 3: Handling inconsistencies (if any)
# Example: Check if Gender column has any inconsistencies and encode it if needed
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
# Step 4: Data normalization or scaling
# Normalize numerical features (Age, Annual Income, Spending Score)

# Normalize numerical columns
normalized_data = data.copy()

# Define normalization function
def normalize_column(column):
    min_val = column.min()
    max_val = column.max()
    return (column - min_val) / (max_val - min_val)

# Apply normalization to numerical columns
normalized_data['Age'] = normalize_column(normalized_data['Age'])
normalized_data['Annual Income (k$)'] = normalize_column(normalized_data['Annual Income (k$)'])
normalized_data['Spending Score (1-100)'] = normalize_column(normalized_data['Spending Score (1-100)'])

# Show normalized data
print(normalized_data.head())
## Exploratory Data Analysis (EDA):
# 1. Distributions
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(data['Age'], bins=20, kde=True, color='skyblue')
plt.title('Age Distribution')

plt.subplot(1, 3, 2)
sns.histplot(data['Annual Income (k$)'], bins=20, kde=True, color='salmon')
plt.title('Annual Income Distribution')

plt.subplot(1, 3, 3)
sns.histplot(data['Spending Score (1-100)'], bins=20, kde=True, color='green')
plt.title('Spending Score Distribution')

plt.tight_layout()
plt.show()


# 2. Correlations (using bar plots)
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
sns.barplot(x='Gender', y='Age', data=data, palette='Set1', estimator='mean')
plt.title('Average Age by Gender')

plt.subplot(1, 2, 2)
sns.barplot(x='Gender', y='Annual Income (k$)', data=data, palette='Set2', estimator='mean')
plt.title('Average Annual Income by Gender')

plt.tight_layout()
plt.show()

#Scatter Plot: Spending Score Versus Age 
fig = px.scatter(data, y='Age', x='Spending Score (1-100)', color='Gender',
                 size='Annual Income (k$)', template='plotly_dark', opacity=0.6,
                 height=600, width=800,
                 title='<b> Spending Score Versus Age')
fig.show()
#Scatter Plots: Age, Annual Income, and Spending Score versus Gender 

plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.scatter(data['Age'], data['Spending Score (1-100)'], c=data['Gender'], cmap='viridis', alpha=0.6)
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.title('Age vs Spending Score by Gender')

plt.subplot(1, 3, 2)
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['Gender'], cmap='viridis', alpha=0.6)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Annual Income vs Spending Score by Gender')

plt.subplot(1, 3, 3)
plt.scatter(data['Age'], data['Annual Income (k$)'], c=data['Gender'], cmap='viridis', alpha=0.6)
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.title('Age vs Annual Income by Gender')

plt.tight_layout()
plt.show()
#Correlation Heatmap 
corr = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
#Kernel Density Estimate (KDE) Plots: Age, Annual Income, and Spending Score versus Gender 

fig, axs = plt.subplots(1, 3, figsize=(18, 6))
sns.kdeplot(data=data, x='Age', hue='Gender', ax=axs[0])
axs[0].set_title('Age distribution by Gender')
sns.kdeplot(data=data, x='Annual Income (k$)', hue='Gender', ax=axs[1])
axs[1].set_title('Annual Income distribution by Gender')
sns.kdeplot(data=data, x='Spending Score (1-100)', hue='Gender', ax=axs[2])
axs[2].set_title('Spending Score distribution by Gender')
plt.tight_layout()
plt.show()
# 3. Statistical Summaries
print("\nSummary statistics of numerical variables:")
print(data.describe())
# # K-medoids Clustering:
# Extract numerical features for clustering
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Determine the optimal number of clusters using the elbow method
distortions = []
K = range(1, 11)
for k in K:
    kmedoids_model = KMedoids(n_clusters=k, random_state=42)
    kmedoids_model.fit(X)
    distortions.append(kmedoids_model.inertia_)

# Plot the elbow curve
plt.plot(K, distortions, 'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()


# Based on the elbow curve, choose the optimal number of clusters
n_clusters = 5  # Choose the number of clusters based on the curve


# Apply K-medoids clustering with the chosen number of clusters
kmedoids = KMedoids(n_clusters=n_clusters, random_state=42)
kmedoids.fit(X)


# Assign each data point to a cluster
data['Cluster'] = kmedoids.labels_

# Visualize the clustering result
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('K-medoids Clustering Result')
plt.show()
# # Hierarchical Clustering:
# Extract numerical features for clustering
X1 = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]


# Compute the linkages using different methods
Z1 = linkage(X1, method='single', metric='euclidean')
Z2 = linkage(X1, method='complete', metric='euclidean')
Z3 = linkage(X1, method='average', metric='euclidean')
Z4 = linkage(X1, method='ward', metric='euclidean')


# Plot the dendrograms for each linkage method
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
dendrogram(Z1)
plt.title('Single')

plt.subplot(2, 2, 2)
dendrogram(Z2)
plt.title('Complete')

plt.subplot(2, 2, 3)
dendrogram(Z3)
plt.title('Average')

plt.subplot(2, 2, 4)
dendrogram(Z4)
plt.title('Ward')

plt.tight_layout()
plt.show()
# #Evaluation and Interpretation
To evaluate the results of each clustering technique individually and collectively, we can assess the quality of the clusters using internal validation metrics and external criteria. Internal validation metrics include measures such as silhouette score and Davies-Bouldin index, which evaluate the compactness and separation of clusters. External criteria may involve comparing the clustering results to known ground truth labels if available.
Let's evaluate the K-medoids clustering and hierarchical clustering results individually using the silhouette score, and then compare and contrast the outcomes from each technique.
# Evaluate K-medoids clustering using silhouette score
silhouette_kmedoids = silhouette_score(X, kmedoids.labels_)
print("Silhouette Score for K-medoids Clustering:", silhouette_kmedoids)

# For hierarchical clustering, we need to specify the number of clusters based on the dendrogram
# Let's assume we chose 5 clusters from the dendrogram

# Apply hierarchical clustering with the chosen number of clusters
hierarchical = AgglomerativeClustering(n_clusters=5)
hierarchical.fit(X)

# Evaluate hierarchical clustering using silhouette score
silhouette_hierarchical = silhouette_score(X, hierarchical.labels_)
print("Silhouette Score for Hierarchical Clustering:", silhouette_hierarchical)
After obtaining silhouette scores for both clustering techniques, we can compare and contrast their outcomes. A higher silhouette score indicates better-defined clusters with greater separation between clusters. Additionally, we can visually inspect the clustering results to gain insights into the dataset's underlying patterns and structures.
# Visualize K-medoids clustering result
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('K-medoids Clustering Result')
plt.show()
# Visualize Hierarchical clustering result
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=hierarchical.labels_, cmap='viridis')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Hierarchical Clustering Result')
plt.show()
By comparing the silhouette scores and visually inspecting the clustering results, we can assess the quality of clusters and gain a comprehensive understanding of the dataset's underlying patterns and structures. We can also consider external criteria if available, such as domain knowledge or known ground truth labels, to further evaluate the clustering outcomes.
