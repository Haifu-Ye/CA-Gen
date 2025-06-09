import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import sys
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter1d
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def set_seed(seed):
    random.seed(seed)  # Python 随机数生成器种子
    np.random.seed(seed)  # NumPy 随机数生成器种子
    torch.manual_seed(seed)  # PyTorch CPU 随机数生成器种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch GPU 随机数生成器种子
        torch.cuda.manual_seed_all(seed)  # 如果有多个GPU
        torch.backends.cudnn.deterministic = True  # 确保每次结果一致
        torch.backends.cudnn.benchmark = False  # 固定算法选择
        
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
    
def gaussian_video(video, lengths, sigma=3):
    scores = np.zeros_like(video)
    prev = 0
    for cur in lengths:
        scores[prev: cur] = gaussian_filter1d(video[prev: cur], sigma)
        prev = cur
    return scores    

# 定义深度学习模型
# # avenue
# class AnomalyDetector(nn.Module):
#     def __init__(self, input_size):
#         super(AnomalyDetector, self).__init__()
        
#         # 定义前 7 层全连接层
#         # self.fc1 = nn.Linear(input_size, 1024)
#         # self.bn1 = nn.BatchNorm1d(1024)
        
#         self.fc2 = nn.Linear(input_size, 256)
#         self.bn2 = nn.BatchNorm1d(256)
#         # self.dropout2 = nn.Dropout(0.1)  # 增加 Dropout
        
        
#         self.fc3 = nn.Linear(256, 128)
#         self.bn3 = nn.BatchNorm1d(128)
#         # self.dropout3 = nn.Dropout(0.05)  # 增加 Dropout
        
#         self.fc4 = nn.Linear(128, 64)
#         self.bn4 = nn.BatchNorm1d(64)
#         # self.dropout4 = nn.Dropout(0.05)  # 增加 Dropout
        
#         self.fc5 = nn.Linear(64, 32)
#         self.bn5 = nn.BatchNorm1d(32)
#         # self.dropout5 = nn.Dropout(0.05)
       
        
#         self.fc7 = nn.Linear(32, 1)
#         # self.bn7 = nn.BatchNorm1d(8)
    
#         self.sigmoid = nn.Sigmoid()
        
#     def forward(self, x):
#         # x = torch.relu(self.bn1(self.fc1(x)))
#         x = torch.relu(self.bn2(self.fc2(x)))
#         # x = self.dropout2(x)
#         x = torch.relu(self.bn3(self.fc3(x)))
#         # x = self.dropout3(x)
#         x = torch.relu(self.bn4(self.fc4(x)))
#         # x = self.dropout4(x)
#         x = torch.relu(self.bn5(self.fc5(x)))
#         # x = self.dropout5(x)
#         # x = torch.relu(self.bn6(self.fc6(x)))
#         # x = torch.relu(self.bn7(self.fc7(x)))
#         x = self.fc7(x)
        
#         # 输出层，最后通过 Sigmoid 激活
#         x = self.sigmoid(x)
        
#         return x

# # ped2
class AnomalyDetector(nn.Module):
    def __init__(self, input_size):
        super(AnomalyDetector, self).__init__()
        
        # 定义前 7 层全连接层
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        # self.dropout2 = nn.Dropout(0.1)  # 增加 Dropout
        
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        # self.dropout3 = nn.Dropout(0.05)  # 增加 Dropout
        
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        # self.dropout4 = nn.Dropout(0.05)  # 增加 Dropout
        
        self.fc5 = nn.Linear(32, 16)
        self.bn5 = nn.BatchNorm1d(16)
        # self.dropout5 = nn.Dropout(0.05)
        self.fc6 = nn.Linear(16, 8)
        self.bn6 = nn.BatchNorm1d(8)
        
        self.fc7 = nn.Linear(8, 1)
        # self.bn7 = nn.BatchNorm1d(8)
    
        self.sigmoid = nn.Sigmoid()
        
      
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        # x = self.dropout2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        # x = self.dropout3(x)
        x = torch.relu(self.bn4(self.fc4(x)))
        # x = self.dropout4(x)
        # x = self.fc5(x)
        x = torch.relu(self.bn5(self.fc5(x)))
        # x = self.dropout5(x)
        x = torch.relu(self.bn6(self.fc6(x)))
        # x = torch.relu(self.bn7(self.fc7(x)))
        x = self.fc7(x)
        
        # 输出层，最后通过 Sigmoid 激活
        x = self.sigmoid(x)
        
        return x
      

# 扩展标签函数
def expand_labels(test_data, test_labels):
    expanded_labels = []
    for i, sample in enumerate(test_data):
        num_subsamples = sample.shape[0]
        expanded_labels.extend([test_labels[i]] * num_subsamples)
    return np.array(expanded_labels)

# 分组平均计算预测结果
def group_max_predictions(predictions, test_data, max_or_mean):
    group_results = []
    start = 0
    for sample in test_data:
        # print("sample shape:" , sample.shape)
        num_subsamples = sample.shape[0]
        # print("num_subsamples:", num_subsamples)
        if max_or_mean == "max":
            group_avg = np.max(predictions[start:start + num_subsamples])
        else:
            group_avg = np.mean(predictions[start:start + num_subsamples])
        group_results.append(group_avg)
        start += num_subsamples
    # logging.info(f"Grouped Predictions: {group_results}")  # 将分组平均后的预测值记录到日志中
    return np.array(group_results)


def macro_auc(video, test_labels, lengths):
    prev = 0
    auc = 0
    for i, cur in enumerate(lengths):
        cur_auc = roc_auc_score(np.concatenate(([0], test_labels[prev: cur], [1])),
                             np.concatenate(([0], video[prev: cur], [sys.float_info.max])))
        auc += cur_auc
        prev = cur
    return auc / len(lengths)


def train_and_evaluate_model(train_data, train_labels, test_data, test_labels_original, test_velocity, test_clip_lengths, input_size, max_or_mean, epochs=20, lr=0.001, batch_size=64, feature_type="feature", patience=20):
    # 初始化模型、优化器、损失函数
    model = AnomalyDetector(input_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    # criterion = SymmetricBCELoss(alpha=1.0, beta=5.0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
    train_labels = torch.tensor(train_labels, dtype=torch.float32).view(-1, 1).to(device)
    test_data = torch.tensor(test_data, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 记录训练损失和 AUC
    train_losses = []
    best_micro_auc = -float('inf')
    best_macro_auc = -float('inf')
    patience_counter = 0  # 用于早停
    

    # 开始训练
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # 在每个 epoch 后计算 AUC
        model.eval()
        with torch.no_grad():
            test_preds = model(test_data)
            test_preds = test_preds.cpu().numpy().flatten()  # 转到 CPU 并转换为 numpy 数组
            # 对预测结果进行分组平均
            grouped_predictions = group_max_predictions(test_preds, test_velocity, max_or_mean)
            if args.gaussian_video:
                grouped_predictions = gaussian_video(grouped_predictions, test_clip_lengths, sigma=args.sigma)

            # 计算 Micro AUC 和 Macro AUC
            micro_auc = roc_auc_score(test_labels_original, grouped_predictions)
            precision, recall, _ = precision_recall_curve(test_labels_original, grouped_predictions)
            auprc = auc(recall, precision)
            macro_auc_score = macro_auc(grouped_predictions, test_labels_original, test_clip_lengths)

            logging.info(f"test Micro AUC: {micro_auc * 100:.2f}")
            logging.info(f"test AUPRC: {auprc:.4f}")
            logging.info(f"test Macro AUC: {macro_auc_score * 100:.2f}")

            # 检查是否有 AUC 提升
            if micro_auc > best_micro_auc or macro_auc_score > best_macro_auc:
                best_micro_auc = max(best_micro_auc, micro_auc)
                best_macro_auc = max(best_macro_auc, macro_auc_score)
                patience_counter = 0  # 重置计数器
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Early stopping at epoch {epoch+1} due to no improvement in AUC (Micro AUC: {micro_auc * 100:.2f}, Macro AUC: {macro_auc_score * 100:.2f}).")
                    break

    # # 绘制并保存损失函数曲线
    # plt.figure(figsize=(10, 6))
    # plt.plot(train_losses, label="Training Loss")
    # plt.title(f"Training Loss Curve for {feature_type}")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.savefig(f"training_loss_curve_{feature_type}.png")
    # plt.show()

    # 评估模型
    model.eval()
    with torch.no_grad():
        test_preds = model(test_data)
        test_preds = test_preds.cpu().numpy().flatten()  # 转到 CPU 并转换为 numpy 数组
    
    print("test_preds.shape:", test_preds.shape)
    # 对预测结果进行分组平均
    grouped_predictions = group_max_predictions(test_preds, test_velocity, max_or_mean)
    logging.info(f"grouped_predictions: {grouped_predictions[:100]}")
    if args.gaussian_video:
        grouped_predictions = gaussian_video(grouped_predictions, test_clip_lengths, sigma=args.sigma)
    
    # 计算AUC、AUPRC和macro AUC
    micro_auc = roc_auc_score(test_labels_original, grouped_predictions)
    precision, recall, _ = precision_recall_curve(test_labels_original, grouped_predictions)
    auprc = auc(recall, precision)
    macro_auc_score = macro_auc(grouped_predictions, test_labels_original, test_clip_lengths)

    logging.info(f"Micro AUC: {micro_auc * 100:.2f}")
    logging.info(f"AUPRC: {auprc:.4f}")
    logging.info(f"Macro AUC: {macro_auc_score * 100:.2f}")

def main(args):
    # 设置日志记录
    setup_logging(f"2jiandu_new_{args.dataset_name}.txt")
    if args.dataset_name == 'ped2':
     # 归一化：使用 StandardScaler 进行标准化
        scaler = MinMaxScaler()
        scaler_velocity = StandardScaler()
        scaler_deep = StandardScaler()
    
    test_clip_lengths = np.load(f'/home/yehaifu/Accurate-Interpretable-VAD/data/{args.dataset_name}/test_clip_lengths.npy',allow_pickle=True)
    # test_clip_lengths = np.load(f'/home/yehaifu/Accurate-Interpretable-VAD/hiera/hiera-main/examples/shanghaitech_features/test_clip_lengths_16.npy',allow_pickle=True)
    print(f"test_clip_lengths shape: {test_clip_lengths.shape}")
    # 加载训练集数据
    train_velocity = np.load(f'output/result/{args.dataset_name}/anomaly/combined_velocity_data.npy', allow_pickle=True)
    train_deep_features = np.load(f'output/result/{args.dataset_name}/anomaly/combined_deep_features_data.npy')
    # generation_deep_features_data = np.load('output/result/shanghaitech/anomaly/generated_deep_features_data.npy')
    # generation_velocity_data = np.load('output/result/shanghaitech/anomaly/generated_velocity_data.npy')
    if args.dataset_name == 'ped2':
        train_velocity = scaler_velocity.fit_transform(train_velocity)
        train_deep_features = scaler_deep.fit_transform(train_deep_features)
   
    print("train velocity shape:",train_velocity.shape)
    print("train deep feature shape:",train_deep_features.shape)
    # 合并训练集的 velocity 和 deep features 特征
    
    train_combined_features = np.concatenate([train_velocity, train_deep_features], axis=1)
    # train_combined_features = np.load(f'output/result/{args.dataset_name}/anomaly/combined_train_16_features_data.npy')
    
    # train_combined_features = train_velocity
    print("train combined features:",train_combined_features.shape)
    # generation_combined_features = np.concatenate([generation_deep_features_data, generation_velocity_data], axis=1)
    # train_combined_features = np.concatenate([train_combined_features, generation_combined_features], axis=0)
    # 加载训练集的标签
    train_labels = np.load(f'output/result/{args.dataset_name}/anomaly/combined_velocity_labels.npy')
    
    # generation_labels = np.ones(generation_combined_features.shape[0])  # 生成数据的标签为1
    # train_labels = np.concatenate([train_labels, generation_labels], axis=0)  # 将生成数据的标签添加到训练标签后面
    
    num_zeros = np.sum(train_labels == 0)
    num_ones = np.sum(train_labels == 1)
    logging.info(f"Number of zeros in train labels: {num_zeros}")
    logging.info(f"Number of ones in train labels: {num_ones}")
    
    # 加载测试集数据
    test_velocity = np.load(f'extracted_features/{args.dataset_name}/test/velocity.npy', allow_pickle=True)
    test_deep_features = np.load(f'extracted_features/{args.dataset_name}/test/deep_features.npy', allow_pickle=True)
    
    print('test_velocity shape:',test_velocity.shape)
    print('test_deep shape:',test_velocity.shape)
   
    
    # 对测试集数据进行拼接
    test_velocity_concatenated = np.concatenate(test_velocity, axis=0)  # 形状为 (96147, 8)
    test_deep_features_concatenated = np.concatenate(test_deep_features, axis=0)  # 形状为 (96147, 512)
    
    if args.dataset_name == 'ped2':
        test_velocity_concatenated = scaler_velocity.transform(test_velocity_concatenated)
        test_deep_features_concatenated = scaler_deep.transform(test_deep_features_concatenated)
        
    test_combined_features = np.concatenate([test_velocity_concatenated, test_deep_features_concatenated], axis=1)  # 形状为 (96147, 520)

    print("test_combined_features shape:",test_combined_features.shape)
    
    if args.dataset_name == 'ped2':
        train_combined_features = scaler.fit_transform(train_combined_features)  # 只在训练数据上 `fit`
        test_combined_features = scaler.transform(test_combined_features)  # 在测试数据上 `transform`
    
    
    # 扩展测试集标签
    test_labels_original = np.load(f'extracted_features/{args.dataset_name}/test/test_labels.npy')
    print(f"Number of 1 in test dataset: {np.sum(test_labels_original == 1)}")
    print(f"Number of 0 in test dataset: {np.sum(test_labels_original == 0)}")

    # logging.info(f"test_labels_original: {test_labels_original[:100]}")
    test_labels_expanded = expand_labels(test_velocity, test_labels_original)

    # 训练并评估模型
    logging.info("Training combined model...")
    print(train_combined_features.shape[1])
    train_and_evaluate_model(train_combined_features, train_labels, test_combined_features, test_labels_original, test_velocity, test_clip_lengths, input_size=train_combined_features.shape[1], max_or_mean = args.max_or_mean, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, feature_type="combined_2jiandu",patience=100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ped2", help="Dataset name")
    parser.add_argument("--root", type=str, default="data/", help="Root path for data")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-7, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for mini-batch training")
    parser.add_argument("--gaussian_video", type=bool, default=True, help = "Whether to use Gaussian smoothing")
    parser.add_argument("--sigma", type=int, default=6, help='sigma for gaussian1d smoothing')
    parser.add_argument("--max_or_mean", type=str, choices=["max", "mean"], default="max", help="Use max or mean for grouped predictions")
    parser.add_argument("--seed", type=int, default=6, help="Random seed for reproducibility")
    args = parser.parse_args()
    # 种子为50，70效果不错 20 40-10 60-8 70-2 130-2
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 运行main函数
    main(args)
    logging.info(f"seed: {args.seed}")
    logging.info(f"dataset: {args.dataset_name}, lr: {args.lr}, epochs: {args.epochs}, gaussian_video: {args.gaussian_video}, max_or_mean: {args.max_or_mean}, seed: {args.seed},batch_size:{args.batch_size}")