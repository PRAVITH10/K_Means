# 🎯 K-Means Clustering Machine Learning Project

A simple and practical implementation of K-Means Clustering for grouping similar data points.

## 🤔 What is K-Means Clustering?

K-Means is like **automatically sorting items into groups**. Imagine organizing students into study groups based on their study hours and grades - K-Means finds these natural groupings without you telling it what each group should look like!

### ✨ Why Use K-Means?

| Advantage | Description |
|-----------|-------------|
| ⚡ **Fast** | Works quickly even with large datasets |
| 🎯 **Simple** | Easy to understand and implement |
| 💡 **Unsupervised** | No labeled data needed |
| 🔧 **Versatile** | Works for many problems |

## 🚀 Quick Start

### Simple Example - Customer Grouping
```python
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Sample data: customer income and spending
data = {
    'income': [15, 16, 40, 42, 60, 62],
    'spending': [39, 81, 76, 94, 17, 81]
}

df = pd.DataFrame(data)
X = df[['income', 'spending']]

# Create 3 groups
kmeans = KMeans(n_clusters=3, random_state=42)
df['group'] = kmeans.fit_predict(X)

# Visualize
plt.scatter(df['income'], df['spending'], c=df['group'], cmap='viridis')
plt.xlabel('Income')
plt.ylabel('Spending')
plt.title('Customer Groups')
plt.show()
```

## 🎛️ Key Parameters

```python
KMeans(
    n_clusters=3,         # Number of groups you want
    random_state=42       # For consistent results
)
```

## 🔍 Finding the Best Number of Clusters

### Method 1: Elbow Method (Inertia)
**Inertia** measures how tightly grouped the clusters are. Lower is better!

```python
from sklearn.cluster import KMeans

# Try different numbers of clusters
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    print(f"k={k}: Inertia = {kmeans.inertia_:.2f}")

# Plot to find the "elbow"
plt.plot(range(1, 11), inertias, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method - Find the Bend!')
plt.grid(True)
plt.show()
```

### Method 2: Silhouette Score
**Silhouette Score** measures how well-separated clusters are. Score ranges from -1 to 1, higher is better!

```python
from sklearn.metrics import silhouette_score

# Calculate silhouette scores
silhouette_scores = []
for k in range(2, 11):  # Need at least 2 clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    print(f"k={k}: Silhouette Score = {score:.3f}")

# Plot silhouette scores
plt.plot(range(2, 11), silhouette_scores, 'go-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score - Higher is Better!')
plt.grid(True)
plt.show()
```

### Combined Approach
```python
import numpy as np

# Compare both methods
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Inertia plot
ax1.plot(range(1, 11), inertias, 'bo-')
ax1.set_xlabel('Number of Clusters')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method')
ax1.grid(True)

# Silhouette plot
ax2.plot(range(2, 11), silhouette_scores, 'go-')
ax2.set_xlabel('Number of Clusters')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis')
ax2.grid(True)

plt.tight_layout()
plt.show()

# Find best k by silhouette score
best_k = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"\n🎯 Optimal number of clusters: {best_k}")
```

## 📊 Understanding Results

```python
# See cluster centers
print("Group Centers:")
print(kmeans.cluster_centers_)

# Check model performance
print(f"\nInertia: {kmeans.inertia_:.2f}")
print(f"Silhouette Score: {silhouette_score(X, df['group']):.3f}")

# Predict new data
new_customer = [[45, 70]]
group = kmeans.predict(new_customer)
print(f"\nNew customer belongs to group {group[0]}")
```

## 📈 What Do These Metrics Mean?

### Inertia
- Sum of squared distances from points to their cluster center
- **Lower = Better** (points are closer to their centers)
- Look for the "elbow" where it stops decreasing rapidly

### Silhouette Score
- Measures how similar a point is to its own cluster vs other clusters
- **Range**: -1 (wrong cluster) to +1 (perfect cluster)
- **Good**: 0.5 - 1.0
- **Fair**: 0.2 - 0.5
- **Poor**: < 0.2

## 🔧 Common Use Cases

- **🛍️ Marketing**: Customer segmentation
- **📷 Images**: Color compression
- **🏥 Healthcare**: Patient grouping
- **📦 Logistics**: Route optimization

## 💡 Pro Tips

### Always Scale Your Data
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
```

### Analyze Each Group
```python
for i in range(3):
    group_data = df[df['group'] == i]
    print(f"Group {i}: {len(group_data)} customers")
    print(f"  Avg Income: {group_data['income'].mean():.1f}")
    print(f"  Avg Spending: {group_data['spending'].mean():.1f}\n")
```

## 🚨 Common Mistakes

- ❌ Not scaling features with different ranges
- ❌ Choosing random number of clusters without validation
- ❌ Ignoring outliers in data
- ❌ Not checking both inertia and silhouette score

## ✅ Best Practices

- ✅ Use `StandardScaler` for different feature scales
- ✅ Use **both** elbow method and silhouette score
- ✅ Set `random_state` for reproducible results
- ✅ Validate clusters make business sense

⭐ **Found this helpful?** Give it a star and share with others!

*Built with ❤️ by [PRAVITH10](https://github.com/PRAVITH10)*
