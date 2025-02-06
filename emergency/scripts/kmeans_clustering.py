# scripts/kmeans_clustering.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle

def cluster_tasks(file_path):
    tasks = pd.read_csv(file_path)
    task_features = tasks[['Task Type', 'Task Duration']]
    task_features = pd.get_dummies(task_features, columns=['Task Type'])

    scaler = StandardScaler()
    task_features_scaled = scaler.fit_transform(task_features)

    kmeans = KMeans(n_clusters=3, random_state=0).fit(task_features_scaled)
    tasks['Cluster'] = kmeans.labels_

    with open('../models/kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)

    tasks.to_csv('../data/clustered_tasks.csv', index=False)
    print("Clustering complete!")

if __name__ == "__main__":
    cluster_tasks("../data/processed_tasks.csv")
