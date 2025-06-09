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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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

# 设置随机种子
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

def plot_clusters(features, labels, centers, anomalies=None, top_k_points=None, title="", save_path=""):
    """
    绘制簇分布图，支持标记簇中心、异常点和簇的 top_k 边缘点。
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

def save_data_and_anomalies(features_dict, anomaly_data_dict, output_dir):
    """
    保存原始数据、生成的异常数据以及标签。
    """
    anomaly_dir = os.path.join(output_dir, 'anomaly')  # 确保将异常文件保存到 anomaly 目录中
    if not os.path.exists(anomaly_dir):
        os.makedirs(anomaly_dir)

    for feature_type, features in features_dict.items():
        if feature_type not in anomaly_data_dict:
            logging.warning(f"No anomaly data for feature type: {feature_type}")
            continue
        
        anomaly_data = anomaly_data_dict[feature_type]

        # 保存异常数据
        anomaly_data_filepath = os.path.join(anomaly_dir, f'{feature_type}_anomaly_data.npy')
        logging.info(f'{feature_type}_anomaly_data.npy shape: {anomaly_data.shape}')
        np.save(anomaly_data_filepath, anomaly_data)
        logging.info(f"Anomaly data for {feature_type} saved to {anomaly_data_filepath}")
        
        # 合并原始数据和异常数据
        combined_data = np.concatenate([features, anomaly_data], axis=0)
        combined_data_filepath = os.path.join(anomaly_dir, f'combined_{feature_type}_data.npy')
        logging.info(f'combined_{feature_type}_data.npy shape: {combined_data.shape}')
        np.save(combined_data_filepath, combined_data)
        logging.info(f"Combined data for {feature_type} saved to {combined_data_filepath}")

        # 生成并保存标签
        original_labels = np.zeros(features.shape[0])  # 原始数据的标签为 0（正常）
        anomaly_labels = np.ones(anomaly_data.shape[0])  # 异常数据的标签为 1（异常）
        combined_labels = np.concatenate([original_labels, anomaly_labels], axis=0)
        combined_labels_filepath = os.path.join(anomaly_dir, f'combined_{feature_type}_labels.npy')
        np.save(combined_labels_filepath, combined_labels)
        logging.info(f"Combined labels for {feature_type} saved to {combined_labels_filepath}")

# 定义 PyTorch 数据集
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# 定义 PyTorch 数据集
class AnomalyDataset(Dataset):
    def __init__(self, h0, h):# 将h0作为输入，normal和h作为目标异常数据
        self.h0 = torch.tensor(h0, dtype=torch.float32)
        self.h = torch.tensor(h, dtype=torch.float32)
    def __len__(self):
        return self.h.size(0)  # 返回数据集的样本数量
    
    def __getitem__(self, idx):
        # 返回随机向量 h0、正常/异常数据以及标签
        return self.h0[idx], self.h[idx]

# 定义 MLP 模型
class MLPGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLPGenerator, self).__init__()
        # 生成器
        self.generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(hidden_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim / 2), input_dim),
            nn.Tanh()  # 生成的h1，值范围为 [-1, 1]，视具体数据需求而定
        )

    def forward(self, h0):
        h1 = self.generator(h0)  # 生成与异常数据相似的样本
        return h1


# 定义训练函数
def train_mlp_generator(h, normal, input_dim, hidden_dim=128, num_epochs=100, batch_size=64, learning_rate=1e-3, lambda_mse=0.5, device='cpu'):
    """
    训练 MLP 生成器生成与 h 相似的 h1。
    
    参数:
    -------
    h: numpy array of shape (N, D), target anomaly data
    input_dim: int, 输入和输出的维度
    hidden_dim: int, 隐藏层维度
    num_epochs: int, 训练轮数
    batch_size: int, 批大小
    learning_rate: float, 学习率
    lambda_mse: float, MSE 损失的权重
    device: str, 'cpu' 或 'cuda'
    
    返回:
    -------
    model: 训练好的 MLPGenerator 模型
    h1: 生成的异常数据，numpy array of shape (N, D)
    """
    # 初始化随机向量 h0
    h0 = np.random.normal(-1, 1, size=h.shape).astype(np.float32)
    dataset = AnomalyDataset(h0, h) # (h,h1,normal)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MLPGenerator(input_dim=input_dim, hidden_dim=hidden_dim).to(device)

    criterion_mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_h0, batch_h in dataloader:
            batch_h0 = batch_h0.to(device)
            batch_h = batch_h.to(device)

            optimizer.zero_grad()
            generated_h1 = model(batch_h0)

            # MSE 损失
            loss_mse = criterion_mse(generated_h1, batch_h)

            # 总损失
            loss = loss_mse
            # loss = loss_mse
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_h0.size(0)
        
        epoch_loss /= len(dataloader.dataset)
        # if (epoch+1) % 10 == 0 or epoch == 0:
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # 生成 h1
    model.eval()
    with torch.no_grad():
        h0_tensor = torch.tensor(h0, dtype=torch.float32).to(device)
        h1_tensor = model(h0_tensor)
        h1 = h1_tensor.cpu().numpy()

    return model, h1

def main(args):
    setup_logging("ananoly_mlp_generation_log.txt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    train_deep_features = np.load('extracted_features/{}/train/deep_features.npy'.format(args.dataset_name), allow_pickle=True)
    train_velocity = np.load('extracted_features/{}/train/velocity.npy'.format(args.dataset_name), allow_pickle=True)
    train_velocity = np.concatenate(train_velocity, 0)
    train_deep_features = np.concatenate(train_deep_features, 0)
    # 加载特征数据
    feature_types = ['deep_features', 'velocity']
    features_dict = {}
    anomaly_data_dict = {}
    
    for feature_type in feature_types:
        feature_filename = f'extracted_features/{args.dataset_name}/train/{feature_type}.npy'
        features = np.load(feature_filename, allow_pickle=True)
        logging.info(f"Loaded {feature_type} with shape: {features.shape}")
        
        # 加载聚类标签和中心
        labels, centers = load_clustering_results(args.output_dir, args.n_components, feature_type)
        logging.info(f"Loaded clustering results for {feature_type}")
        
        # 预处理特征数据
        features, _ = preprocess_data(features, labels)
        logging.info(f"Preprocessed {feature_type} with shape: {features.shape}")
        
        features_dict[feature_type] = features
        
        # 加载之前生成的异常数据 h
        anomaly_data_filepath = os.path.join(args.output_dir, 'anomaly', f'{feature_type}_anomaly_data.npy')
        if not os.path.exists(anomaly_data_filepath):
            logging.error(f"Anomaly data file not found: {anomaly_data_filepath}")
            continue
        
        h = np.load(anomaly_data_filepath)
        anomaly_data_dict[feature_type] = h
        logging.info(f"Loaded existing anomaly data for {feature_type} with shape: {h.shape}")
    
    # 此处不再保存原始数据和已加载的异常数据
    # save_data_and_anomalies(features_dict, anomaly_data_dict, args.output_dir)
    # logging.info("Saved original and existing anomaly data.")
    
    # 训练 MLP 生成 h1 并拼接
    for feature_type in feature_types:
        if feature_type not in anomaly_data_dict:
            logging.warning(f"Skipping MLP training for {feature_type} due to missing anomaly data.")
            continue
        
        h = anomaly_data_dict[feature_type]
        normal = features_dict[feature_type]
        input_dim = h.shape[1]
        logging.info(f"Training MLP for {feature_type} with input dimension: {input_dim}")
        
        if feature_type == 'deep_features':
            model, h1 = train_mlp_generator(
                h=h,
                normal = normal,
                input_dim=512,
                hidden_dim=256,
                num_epochs=20,
                batch_size=args.mlp_batch_size,
                learning_rate=1e-6,
                lambda_mse=args.mlp_lambda_mse,
                device=device
            )
            logging.info(f"Generated h1 for {feature_type} with shape: {h1.shape}")
        # hidden_dim原来是256
        
        
        if feature_type == 'velocity':
            model, h1 = train_mlp_generator(
                h=h,
                normal = normal,
                input_dim=1,
                hidden_dim=4,
                num_epochs=20,
                batch_size=args.mlp_batch_size,
                learning_rate=1e-6,
                lambda_mse=args.mlp_lambda_mse,
                device=device
            )
            logging.info(f"Generated h1 for {feature_type} with shape: {h1.shape}")
            
        # 拼接 h 和 h1
        final_anomaly_data = np.concatenate([h, h1], axis=0)
        anomaly_data_dict[feature_type] = final_anomaly_data
        logging.info(f"Final anomaly data for {feature_type} with shape: {final_anomaly_data.shape}")
    
    # 保存拼接后的数据
    save_data_and_anomalies(features_dict, anomaly_data_dict, args.output_dir)
    logging.info("Saved augmented anomaly data with h and h1.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ped2", help="Dataset name")
    parser.add_argument("--output_dir", type=str, default="output/result/ped2", help="Output directory for saving results")
    parser.add_argument("--n_components", type=int, default=4, help="Number of clusters for KMeans")
    parser.add_argument("--distance_metric", type=str, default="euclidean", choices=["euclidean", "cityblock", "cosine", "chebyshev","minkowski"], help="Distance metric for anomaly generation")
    parser.add_argument("--seed", type=int, default=40, help="Random seed for reproducibility")
    
    # 新增的训练参数
    parser.add_argument("--mlp_hidden_dim", type=int, default=128, help="Hidden layer dimension for MLP")
    parser.add_argument("--mlp_num_epochs", type=int, default=100, help="Number of epochs for MLP training")
    parser.add_argument("--mlp_batch_size", type=int, default=64, help="Batch size for MLP training")
    parser.add_argument("--mlp_learning_rate", type=float, default=1e-6, help="Learning rate for MLP training")
    parser.add_argument("--mlp_lambda_mse", type=float, default=0.5, help="Lambda value for MSE loss in MLP training")
    
    
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
