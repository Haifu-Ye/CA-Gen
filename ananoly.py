import numpy as np
import os
import random
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
import argparse
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import logging
from scipy.spatial.distance import cdist
import torch
# 初始化日志系统
def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),  # 输出到文件
            logging.StreamHandler()  # 输出到控制台
        ]
    )
    
def set_seed(seed):
    random.seed(seed)  # Python 随机数生成器种子
    np.random.seed(seed)  # NumPy 随机数生成器种子
    torch.manual_seed(seed)  # PyTorch CPU 随机数生成器种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch GPU 随机数生成器种子
        torch.cuda.manual_seed_all(seed)  # 如果有多个GPU
        torch.backends.cudnn.deterministic = True  # 确保每次结果一致
        torch.backends.cudnn.benchmark = False  # 固定算法选择

def load_clustering_results(output_dir, n_components, feature_type):
    """
    从文件中加载聚类标签和聚类中心
    """
    labels_filename = os.path.join(output_dir, f'kmeans_labels_{n_components}_{feature_type}.npy')
    centers_filename = os.path.join(output_dir, f'kmeans_centers_{n_components}_{feature_type}.npy')

    labels = np.load(labels_filename)
    centers = np.load(centers_filename)

    return labels, centers

def preprocess_data(features, labels=None):
    """
    对特征进行拼接，将所有样本的特征拼接成一个二维数组。
    """
    processed_features = []
    valid_labels = []
    
    for i, feature in enumerate(tqdm(features, desc="Processing features")):
        if len(feature) > 0:  # 确保非空
            processed_features.append(feature)
            if labels is not None:
                valid_labels.append(labels[i])

    # 使用 np.concatenate 将所有样本的特征拼接成二维矩阵
    concatenated_features = np.concatenate(processed_features, axis=0)
    
    if labels is not None:
        return concatenated_features, np.array(valid_labels)
    return concatenated_features

import numpy as np
from sklearn.neighbors import NearestNeighbors


import numpy as np
from scipy.spatial.distance import cdist

def generate_anomalies_via_local_edge_points(
    features, 
    labels, 
    centers, 
    n_components, 
    distance_metric, 
    anomaly_ratio=0.1, 
    top_k=5, 
    top_x=3
):
    """
    通过类内离簇中心最远的若干点生成异常数据，并围绕该点进行噪声采样。
    选取top_k个最远点后，计算这些点到其他簇中心的距离和，选出前top_x个距离和最小的点进行异常生成。
    
    :param features: 特征数组，形状为 (num_samples, num_features)
    :param labels: 标签数组，形状为 (num_samples,)
    :param centers: 簇中心数组，形状为 (n_components, num_features)
    :param n_components: 簇的数量
    :param distance_metric: 距离计算方法，如 'euclidean'
    :param anomaly_ratio: 异常比例，默认为 0.1
    :param top_k: 每个簇选取最远的前 K 个点
    :param top_x: 从 top_k 中选取距离和最小的前 X 个点
    :return: (异常数据数组, 每个簇的 top_x 边缘点列表)
    """
    # 创建每个簇的索引列表
    clusters = {i: np.where(labels == i)[0] for i in range(n_components)}
    
    total_samples = features.shape[0]
    print("total_samples:", total_samples)
    num_anomaly_samples = int(total_samples * anomaly_ratio)
    print(f"Generating {num_anomaly_samples} anomaly samples via noise sampling around selected edge points...")

    anomaly_data = []
    selected_top_points = []  # 用于存储每个簇的 selected top_x 边缘点

    valid_clusters = []  # 存储有效簇的索引（点数 >= top_k）
    for cluster_idx in range(n_components):
        cluster_points = features[clusters[cluster_idx]]
        if len(cluster_points) < top_k:
            print(f"Skipping cluster {cluster_idx} with insufficient points: {len(cluster_points)}")
            continue
        valid_clusters.append(cluster_idx)

    num_valid_clusters = len(valid_clusters)
    if num_valid_clusters == 0:
        print("No valid clusters with enough points to generate anomalies.")
        return np.array(anomaly_data), selected_top_points

    anomalies_per_cluster = num_anomaly_samples // num_valid_clusters
    remaining_anomalies = num_anomaly_samples % num_valid_clusters

    # 确保每个簇的异常点数一致，优先分配剩余异常点
    anomalies_per_cluster += (1 if remaining_anomalies > 0 else 0)

    for cluster_idx in valid_clusters:
        cluster_points = features[clusters[cluster_idx]]
        center = centers[cluster_idx]

        # 计算每个点到簇中心的距离
        local_distances = cdist(cluster_points, center[None, :], metric=distance_metric).flatten()

        # 计算当前簇的距离标准差
        distance_std = local_distances.std()
        print(f"Cluster {cluster_idx}: Mean distance = {local_distances.mean():.4f}, Std deviation = {distance_std:.4f}")

        # 找到距离簇中心最远的前 top_k 个点
        top_k_indices = np.argsort(local_distances)[-top_k:]
        edge_points = cluster_points[top_k_indices]
        print(f"Cluster {cluster_idx}: Selected top_k edge points shape: {edge_points.shape}")

        # 计算每个 edge_point 到其他簇中心的距离和
        other_centers = np.delete(centers, cluster_idx, axis=0)  # 移除当前簇的中心
        sum_distances = cdist(edge_points, other_centers, metric=distance_metric).sum(axis=1)

        # 选择 sum_distances 最小的前 top_x 个点
        if top_x > top_k:
            print(f"top_x ({top_x}) is greater than top_k ({top_k}), reducing top_x to top_k.")
            current_top_x = top_k
        else:
            current_top_x = top_x

        top_x_indices = np.argsort(sum_distances)[:current_top_x]
        selected_points = edge_points[top_x_indices]
        print(f"Cluster {cluster_idx}: Selected top_x points shape: {selected_points.shape}")
        selected_top_points.append(selected_points)

        # 分配每个 selected_point 生成的异常数量
        anomalies_per_selected = anomalies_per_cluster // current_top_x
        remaining_anomalies_in_cluster = anomalies_per_cluster % current_top_x

        anomalies_generated = 0
        for idx, edge_point in enumerate(selected_points):
            points_to_generate = anomalies_per_selected + (1 if idx < remaining_anomalies_in_cluster else 0)
            for _ in range(points_to_generate):
                # 计算噪声的范围，确保噪声不会超出 edge_point 的范围
                min_val, max_val = edge_point.min(), edge_point.max()
                noise_range = max_val - min_val

                # 生成噪声并限制在 [min_val, max_val] 之间
                noise = np.random.normal(loc=0, scale=distance_std, size=edge_point.shape)
                anomaly_point = edge_point + noise
                anomaly_data.append(anomaly_point)
                anomalies_generated += 1

        # 如果因整除不足而需要补足异常点
        while anomalies_generated < anomalies_per_cluster:
            edge_point = selected_points[np.random.randint(len(selected_points))]
            min_val, max_val = edge_point.min(), edge_point.max()
            noise_range = max_val - min_val

            # 生成噪声并限制在 [min_val, max_val] 之间
            noise = np.random.normal(loc=0, scale=distance_std, size=edge_point.shape)
            anomaly_point = edge_point + noise
            anomaly_data.append(anomaly_point)
            anomalies_generated += 1

    # 校正异常点数量
    generated_anomalies = len(anomaly_data)
    if generated_anomalies != num_anomaly_samples:
        anomaly_data = anomaly_data[:num_anomaly_samples]  # 截断或补充异常点

    return np.array(anomaly_data), selected_top_points


def save_data_and_anomalies(features_dict, anomaly_data_dict, output_dir):
    """
    保存原始数据、生成的异常数据以及标签。
    """
    anomaly_dir = os.path.join(output_dir, 'anomaly')  # 确保将异常文件保存到 anomaly 目录中
    if not os.path.exists(anomaly_dir):
        os.makedirs(anomaly_dir)

    for feature_type, features in features_dict.items():
        anomaly_data = anomaly_data_dict[feature_type]

        # 保存异常数据
        anomaly_data_filepath = os.path.join(anomaly_dir, f'{feature_type}_anomaly_data.npy')
        print(f'{feature_type}_anomaly_data.npy shape', anomaly_data.shape)
        np.save(anomaly_data_filepath, anomaly_data)
        print(f"Anomaly data for {feature_type} saved to {anomaly_data_filepath}")
        
        # 合并原始数据和异常数据
        combined_data = np.concatenate([features, anomaly_data], axis=0)
        combined_data_filepath = os.path.join(anomaly_dir, f'combined_{feature_type}_data.npy')
        print(f'combined_{feature_type}_data.npy shape:',combined_data.shape)
        np.save(combined_data_filepath, combined_data)
        print(f"Combined data for {feature_type} saved to {combined_data_filepath}")

        # 生成并保存标签
        original_labels = np.zeros(features.shape[0])  # 原始数据的标签为 0（正常）
        anomaly_labels = np.ones(anomaly_data.shape[0])  # 异常数据的标签为 1（异常）
        combined_labels = np.concatenate([original_labels, anomaly_labels], axis=0)
        combined_labels_filepath = os.path.join(anomaly_dir, f'combined_{feature_type}_labels.npy')
        np.save(combined_labels_filepath, combined_labels)
        print(f"Combined labels for {feature_type} saved to {combined_labels_filepath}")

def plot_clusters(features, labels, centers, anomalies=None, top_k_points=None, title="", save_path=""):
    """
    绘制簇分布图，支持标记簇中心、异常点和簇的 top_k 边缘点。
    :param features: 原始特征数据
    :param labels: 每个点所属的簇标签
    :param centers: 每个簇的中心点
    :param anomalies: 异常点数据（可选）
    :param top_k_points: 每簇的 top_k 边缘点（列表，长度为簇数）
    :param title: 图标题
    :param save_path: 图像保存路径
    """
    # 使用 PCA 将数据降到二维
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    centers_2d = pca.transform(centers)

    if anomalies is not None:
        anomalies_2d = pca.transform(anomalies)

    plt.figure(figsize=(12, 10))

    # 绘制簇内点
    for cluster_idx in np.unique(labels):
        cluster_points = features_2d[labels == cluster_idx]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_idx}", alpha=0.5, s=10)

    # 绘制簇中心
    for idx, (cx, cy) in enumerate(centers_2d):
        plt.scatter(cx, cy, c='black', marker='x', label=f"Center {idx}", s=100)

    # 绘制 top_k 边缘点
    if top_k_points is not None:
        for cluster_idx, cluster_top_k_points in enumerate(top_k_points):
            if cluster_top_k_points is not None and len(cluster_top_k_points) > 0:
                cluster_top_k_2d = pca.transform(cluster_top_k_points)
                plt.scatter(cluster_top_k_2d[:, 0], cluster_top_k_2d[:, 1], c='black', marker='^', 
                            label=f"Top {len(cluster_top_k_points)} Points (Cluster {cluster_idx})", s=100)

    # 绘制异常点
    if anomalies is not None:
        plt.scatter(anomalies_2d[:, 0], anomalies_2d[:, 1], c='purple', label="Anomalies", alpha=0.05, s=10)

    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    # plt.show()

    
def main(args):
    setup_logging("ananoly_copy_4_log.txt")
    # 加载特征数据
    feature_types = ['deep_features', 'velocity']
    # feature_types = ['train_16_features']
    features_dict = {}
    anomaly_data_dict = {}

    for feature_type in feature_types:
        feature_filename = f'extracted_features/{args.dataset_name}/train/{feature_type}.npy'
        features = np.load(feature_filename, allow_pickle=True)
        print("feature shape:",features.shape)
        
        # 加载聚类标签和中心
        labels, centers = load_clustering_results(args.output_dir, args.n_components, feature_type)

        # 预处理特征数据
        features, _ = preprocess_data(features, labels)
       
        
        print("feature shape:",features.shape)
        
        features_dict[feature_type] = features

        # anomaly_data, top_k_points = generate_anomalies_via_local_edge_points(
        #     features=features,
        #     labels=labels,
        #     centers=centers,
        #     n_components=args.n_components,
        #     distance_metric=args.distance_metric,
        #     # std_dev=1,
        #     anomaly_ratio=args.anomaly_ratio,
        #     top_k=args.top_k,  # 选取每簇前 5 个边缘点
        #     random_seed=args.seed
        # )
        
        anomaly_data, selected_points = generate_anomalies_via_local_edge_points(
            features=features,
            labels=labels,
            centers=centers,
            n_components=args.n_components,
            distance_metric=args.distance_metric,
            anomaly_ratio=args.anomaly_ratio,
            top_k=args.top_k,
            top_x=args.top_x
        )
 
        anomaly_data_dict[feature_type] = anomaly_data  # 确保这里添加了键值对
        
       
    # 保存原始数据、异常数据和标签0
    save_data_and_anomalies(features_dict, anomaly_data_dict, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ped2", help="Dataset name")
    parser.add_argument("--output_dir", type=str, default="output/result/ped2", help="Output directory for saving results")
    parser.add_argument("--n_components", type=int, default=4, help="Number of clusters for KMeans")
    parser.add_argument("--anomaly_ratio", type=float, default=0.5, help="Proportion of total samples to be generated as anomalies")
    parser.add_argument("--distance_metric", type=str, default="euclidean", choices=["euclidean", "cityblock", "cosine", "chebyshev","minkowski"], help="Distance metric for anomaly generation")
    parser.add_argument("--top_k", type=int, default=100, help="top_k of cluster")
    parser.add_argument("--top_x", type=int, default=10, help="top_x of top_k")
    parser.add_argument("--seed", type=int, default=40, help="Random seed for reproducibility")
    # 原来是40

    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
    logging.info(f"top_k:{args.top_k}, top_x:{args.top_x} anomaly_ratio:{args.anomaly_ratio}, distance_metric:{args.distance_metric}")
