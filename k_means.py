import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def preprocess_data(features):
    """
    对数据进行拼接处理，将每个 (x, y) 转换为 (total_samples, y)
    """
    processed_features = []
    for feature in tqdm(features, desc="Processing features"):
        if len(feature) > 0:
            processed_features.append(feature)
    concatenated_features = np.concatenate(processed_features, axis=0)
    print('concatenated_features shape', concatenated_features.shape)
    return concatenated_features

def evaluate_clustering(data, n_components, labels):
    """
    评估聚类效果
    """
    # 计算轮廓系数
    silhouette_avg = silhouette_score(data, labels) if len(np.unique(labels)) > 1 else -1
    
    # 使用肘部法（SSE）
    kmeans = KMeans(n_clusters=n_components, random_state=0)
    kmeans.fit(data)
    sse = kmeans.inertia_
    
    return silhouette_avg, sse

def kmeans_clustering(data, n_components, output_dir='output/result', feature_type='deep'):
    """
    使用 KMeans 对数据进行聚类并保存结果
    """
    # 使用 KMeans 进行聚类
    kmeans = KMeans(n_clusters=n_components, random_state=0)
    kmeans = KMeans(n_clusters=n_components, random_state=0, max_iter=500, n_init=50)
    kmeans.fit(data)
    labels = kmeans.labels_
    
    # dbscan = DBSCAN(eps=0.5, min_samples=5)
    # labels = dbscan.fit_predict(data)

    # 创建输出目录，如果不存在的话
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存聚类标签和模型中心到文件
    labels_filename = os.path.join(output_dir, f'kmeans_labels_{n_components}_{feature_type}.npy')
    np.save(labels_filename, labels)

    centers_filename = os.path.join(output_dir, f'kmeans_centers_{n_components}_{feature_type}.npy')
    np.save(centers_filename, kmeans.cluster_centers_)

    # 评估聚类效果
    # silhouette_avg, sse = evaluate_clustering(data, n_components, labels)

    # 绘制聚类结果
    # plot_clustering_results(data, labels, n_components, output_dir, feature_type)

    # return silhouette_avg, sse,labels
    unique, counts = np.unique(labels, return_counts=True)
    print("每个簇的样本数:")
    for cluster, count in zip(unique, counts):
        print(f"簇 {cluster}: {count} 样本")
        
    return labels

def plot_clustering_results(data, labels, n_components, output_dir, feature_type):
    """
    绘制聚类结果并保存为图像
    """
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)

    plt.figure(figsize=(8, 6))
    for i in range(n_components):
        cluster = data_pca[labels == i]
        plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {i + 1}')

    plt.title(f'KMeans Clustering with {n_components} Clusters ({feature_type})')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()

    # 保存图像到文件
    image_filename = os.path.join(output_dir, f'kmeans_clustering_{n_components}_{feature_type}.png')
    plt.savefig(image_filename)
    plt.close()
    print(f"Cluster plot saved to {image_filename}")

def plot_evaluation_metrics(results, output_dir, feature_type):
    """
    绘制评估指标并保存为图像
    """
    silhouette_dir = os.path.join(output_dir, 'silhouette')
    elbow_dir = os.path.join(output_dir, 'elbow')

    os.makedirs(silhouette_dir, exist_ok=True)
    os.makedirs(elbow_dir, exist_ok=True)

    n_components_range = [result[0] for result in results]
    silhouettes = [result[1] for result in results]
    sses = [result[2] for result in results]

    plt.figure(figsize=(10, 6))
    plt.plot(n_components_range, silhouettes, marker='o', label='Silhouette Score')
    plt.xlabel('Number of clusters')
    plt.ylabel('Score')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(silhouette_dir, f'silhouette_score_{feature_type}.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(n_components_range, sses, marker='o', label='SSE (Elbow Method)')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.title('SSE (Elbow Method) vs Number of Clusters')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(elbow_dir, f'sse_elbow_method_{feature_type}.png'))
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="avenue", help='dataset name')
    parser.add_argument("--n_components_min", type=int, default=4, help='minimum number of KMeans components')
    parser.add_argument("--n_components_max", type=int, default=4, help='maximum number of KMeans components')
    parser.add_argument("--output_dir", type=str, default="output/result/avenue", help='output directory for saving results')
    args = parser.parse_args()

    # 加载并处理三个特征
    for feature_type in ['deep','velocity']:
        feature_filename = f'extracted_features/{args.dataset_name}/train/{feature_type}.npy'
        features = np.load(feature_filename, allow_pickle=True)

        # 直接展开特征（不使用池化）
        processed_features = preprocess_data(features)

        results = []
        for n_components in tqdm(range(args.n_components_min, args.n_components_max + 1), desc=f"Clustering {feature_type}"):
            silhouette_avg, sse, labels = kmeans_clustering(processed_features, n_components=n_components, output_dir=args.output_dir, feature_type=feature_type)
            labels = kmeans_clustering(features, n_components=n_components, output_dir=args.output_dir, feature_type=feature_type)
            results.append((n_components, silhouette_avg, sse))

        plot_evaluation_metrics(results, args.output_dir, feature_type)
        
# import numpy as np
# import os
# from sklearn.cluster import KMeans
# from tqdm import tqdm
# from sklearn.metrics import silhouette_score
# import matplotlib.pyplot as plt
# import argparse

# def preprocess_data(features):
#     """
#     对数据进行拼接处理，将每个 (x, y) 转换为 (total_samples, y)
#     """
#     processed_features = []
#     for feature in tqdm(features, desc="Processing features"):
#         if len(feature) > 0:
#             processed_features.append(feature)
#     concatenated_features = np.concatenate(processed_features, axis=0)
#     print('concatenated_features shape', concatenated_features.shape)
#     return concatenated_features

# def preprocess_and_concatenate_features(deep_features, velocity):
#     """
#     将三个特征分别拼接成二维矩阵，并将它们合并成一个大矩阵。
#     """
#     # 分别拼接每个特征
#     concatenated_deep = preprocess_data(deep_features)
#     concatenated_velocity = preprocess_data(velocity)
#     # concatenated_pose = preprocess_data(aligned_pose)

#     # 将三个特征矩阵拼接在一起，形成 (x, 554) 的形状
#     combined_features = np.concatenate([concatenated_deep, concatenated_velocity], axis=1)
#     print("combined shape:",combined_features.shape)
#     return combined_features

# def kmeans_clustering(data, n_components, output_dir='output/result', feature_type='combined'):
#     """
#     使用 KMeans 对数据进行聚类并保存结果
#     """
#     # 使用 KMeans 进行聚类
#     kmeans = KMeans(n_clusters=n_components, random_state=0)
#     kmeans.fit(data)
#     labels = kmeans.labels_
    
#     # 打印每个簇的样本数
#     print("Cluster sizes:")
#     for i in range(n_components):
#         cluster_size = np.sum(labels == i)
#         print(f"Cluster {i}: {cluster_size} samples")

#     # 创建输出目录，如果不存在的话
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # 保存聚类标签和模型中心到文件
#     labels_filename = os.path.join(output_dir, f'kmeans_labels_{n_components}_{feature_type}.npy')
#     np.save(labels_filename, labels)

#     centers_filename = os.path.join(output_dir, f'kmeans_centers_{n_components}_{feature_type}.npy')
#     np.save(centers_filename, kmeans.cluster_centers_)

#     # 评估聚类效果
#     # silhouette_avg = silhouette_score(data, labels) if len(np.unique(labels)) > 1 else -1
#     # sse = kmeans.inertia_

#     return labels

# def plot_evaluation_metrics(results, output_dir, feature_type):
#     """
#     绘制评估指标并保存为图像
#     """
#     silhouette_dir = os.path.join(output_dir, 'silhouette')
#     elbow_dir = os.path.join(output_dir, 'elbow')

#     os.makedirs(silhouette_dir, exist_ok=True)
#     os.makedirs(elbow_dir, exist_ok=True)

#     n_components_range = [result[0] for result in results]
#     silhouettes = [result[1] for result in results]
#     sses = [result[2] for result in results]

#     plt.figure(figsize=(10, 6))
#     plt.plot(n_components_range, silhouettes, marker='o', label='Silhouette Score')
#     plt.xlabel('Number of clusters')
#     plt.ylabel('Score')
#     plt.title('Silhouette Score vs Number of Clusters')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(os.path.join(silhouette_dir, f'silhouette_score_{feature_type}.png'))
#     plt.close()

#     plt.figure(figsize=(10, 6))
#     plt.plot(n_components_range, sses, marker='o', label='SSE (Elbow Method)')
#     plt.xlabel('Number of clusters')
#     plt.ylabel('SSE')
#     plt.title('SSE (Elbow Method) vs Number of Clusters')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(os.path.join(elbow_dir, f'sse_elbow_method_{feature_type}.png'))
#     plt.close()

# def main(args):
#     # 加载三个特征的数据
#     deep_features = np.load(f'extracted_features/{args.dataset_name}/train/deep_features.npy', allow_pickle=True)
#     velocity = np.load(f'extracted_features/{args.dataset_name}/train/velocity.npy', allow_pickle=True)
#     # aligned_pose = np.load(f'extracted_features/{args.dataset_name}/train/aligned_pose.npy', allow_pickle=True)
#     print("deep shape:",deep_features.shape)
#     print("velocity shape:",velocity.shape)
#     # 将三个特征合并成一个大矩阵
#     combined_features = preprocess_and_concatenate_features(deep_features, velocity)
#     velocity = np.concatenate(velocity, axis=0)
#     print("velocity shape",velocity.shape)
#     # 进行聚类和评价
#     results = []
#     for n_components in tqdm(range(args.n_components_min, args.n_components_max + 1), desc="Clustering velocity features"):
#         labels = kmeans_clustering(velocity, n_components=n_components, output_dir=args.output_dir, feature_type='velocity')
#         # results.append((n_components, silhouette_avg, sse))

#     # 绘制评估指标图
#     # plot_evaluation_metrics(results, args.output_dir, 'combined')


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset_name", type=str, default="shanghaitech", help="Dataset name")
#     parser.add_argument("--n_components_min", type=int, default=4, help="Minimum number of KMeans components")
#     parser.add_argument("--n_components_max", type=int, default=4, help="Maximum number of KMeans components")
#     parser.add_argument("--output_dir", type=str, default="output/result/shanghaitech", help="Output directory for saving results")
#     args = parser.parse_args()

#     main(args)
