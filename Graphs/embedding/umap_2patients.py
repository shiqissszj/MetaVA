import torch
import torch.nn as nn
import pickle
import wfdb
import numpy as np
import umap
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from collections import Counter

# 假设这些是您的模型和数据路径，请根据实际情况修改
models_src = "/Users/shiqissszj/Documents/All_Projects/MetaVa/training_logs/Version2：MAML+CL/trained_models"
data_src = "/Users/shiqissszj/Documents/All_Projects/MetaVa/MetaVA_Codes/MetaDemo_V1/data"

# 加载Net1D模型
def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    print(f"Model loaded from {model_path}")
    return model

# 提取嵌入特征
def extract_embeddings(model, ecg_data):
    model.eval()
    with torch.no_grad():
        ecg_tensor = torch.tensor(ecg_data, dtype=torch.float32).unsqueeze(1)
        # 假设模型有一个 get_embedding 方法
        embedding = model.get_embedding(ecg_tensor)
    return embedding.cpu().numpy() if embedding is not None else None

# 为多位患者并列绘制训练前和训练后的UMAP图
def plot_per_patient_side_by_side(all_initial_embeddings, all_final_embeddings, patient_ids):
    num_patients = len(patient_ids)
    fig, axes = plt.subplots(2, num_patients, figsize=(6 * num_patients, 8))

    for i, pid in enumerate(patient_ids):
        initial_emb = all_initial_embeddings[i]
        final_emb = all_final_embeddings[i]

        # 为该患者单独拟合UMAP模型
        umap_model = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation')
        combined_data = np.concatenate([initial_emb, final_emb], axis=0)
        umap_model.fit(combined_data)

        # 转换嵌入到低维空间
        umap_initial = umap_model.transform(initial_emb)
        umap_final = umap_model.transform(final_emb)

        # 绘制训练前UMAP图（第一行）
        axes[0, i].scatter(umap_initial[:, 0], umap_initial[:, 1],
                           c='#1f77b4', alpha=0.6, marker='o',
                           edgecolors='w', linewidth=0.5)
        axes[0, i].set_title(f'Patient {pid} - Before Training')

        # 绘制训练后UMAP图（第二行）
        axes[1, i].scatter(umap_final[:, 0], umap_final[:, 1],
                           c='#ff7f0e', alpha=0.6, marker='s',
                           edgecolors='k', linewidth=0.5)
        axes[1, i].set_title(f'Patient {pid} - After Training')

        # 统一坐标范围以便比较
        x_min = min(umap_initial[:, 0].min(), umap_final[:, 0].min()) - 0.5
        x_max = max(umap_initial[:, 0].max(), umap_final[:, 0].max()) + 0.5
        y_min = min(umap_initial[:, 1].min(), umap_final[:, 1].min()) - 0.5
        y_max = max(umap_initial[:, 1].max(), umap_final[:, 1].max()) + 0.5

        axes[0, i].set_xlim(x_min, x_max)
        axes[0, i].set_ylim(y_min, y_max)
        axes[1, i].set_xlim(x_min, x_max)
        axes[1, i].set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()

def plot_combined_umap_single_fig(all_initial_embeddings, all_final_embeddings, patient_ids):
    # 将所有患者的训练前和训练后数据合并
    combined_initial = np.concatenate(all_initial_embeddings, axis=0)  # 训练前数据
    combined_final = np.concatenate(all_final_embeddings, axis=0)      # 训练后数据
    combined_all = np.concatenate([combined_initial, combined_final], axis=0)  # 所有数据

    # 使用全局UMAP模型拟合所有数据
    umap_model = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation')
    umap_model.fit(combined_all)

    # 转换训练前和训练后的嵌入到低维空间
    umap_initial = umap_model.transform(combined_initial)
    umap_final = umap_model.transform(combined_final)

    # 为每位患者计算数据点范围
    n_samples_per_patient = [len(embed) for embed in all_initial_embeddings]
    initial_splits = np.split(umap_initial, np.cumsum(n_samples_per_patient)[:-1])
    final_splits = np.split(umap_final, np.cumsum(n_samples_per_patient)[:-1])

    # 创建一个图像，包含两个子图（训练前和训练后）
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 绘制训练前的UMAP图（左侧）
    for i, (patient_emb, pid) in enumerate(zip(initial_splits, patient_ids)):
        axes[0].scatter(patient_emb[:, 0], patient_emb[:, 1],
                        c=f'C{i}', alpha=0.6, marker='o',)
                        #label=f'Patient {pid}')
    axes[0].set_title('Before Training')
    axes[0].legend()

    # 绘制训练后的UMAP图（右侧）
    for i, (patient_emb, pid) in enumerate(zip(final_splits, patient_ids)):
        axes[1].scatter(patient_emb[:, 0], patient_emb[:, 1],
                        c=f'C{i}', alpha=0.6, marker='s',)
                        # label=f'Patient {pid}')
    axes[1].set_title('After Training')
    axes[1].legend()

    # 设置统一的坐标范围
    x_min = min(umap_initial[:, 0].min(), umap_final[:, 0].min()) - 0.5
    x_max = max(umap_initial[:, 0].max(), umap_final[:, 0].max()) + 0.5
    y_min = min(umap_initial[:, 1].min(), umap_final[:, 1].min()) - 0.5
    y_max = max(umap_initial[:, 1].max(), umap_final[:, 1].max()) + 0.5

    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(y_min, y_max)
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()


def plot_overlapped_umap(all_initial_embeddings, all_final_embeddings, patient_ids):
    # 创建一个图像，包含两个子图（训练前和训练后）
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 为每位患者独立拟合 UMAP 并绘制训练前嵌入图
    for i, pid in enumerate(patient_ids):
        # 训练前的 UMAP
        umap_model_initial = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation')
        umap_initial = umap_model_initial.fit_transform(all_initial_embeddings[i])
        axes[0].scatter(umap_initial[:, 0], umap_initial[:, 1],
                        c=f'C{i}', alpha=0.6, marker='o',) #label=f'Patient {pid}')

    axes[0].set_title('Before Training')
    axes[0].legend()

    # 为每位患者独立拟合 UMAP 并绘制训练后嵌入图
    for i, pid in enumerate(patient_ids):
        # 训练后的 UMAP
        umap_model_final = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation')
        umap_final = umap_model_final.fit_transform(all_final_embeddings[i])
        axes[1].scatter(umap_final[:, 0], umap_final[:, 1],
                        c=f'C{i}', alpha=0.6, marker='s',) #label=f'Patient {pid}')

    axes[1].set_title('After Training')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

# 主程序
def main():
    # 加载初始的Net1D模型
    net1d_model_path = models_src + '/raw.pkl'
    net1d_model = load_model(net1d_model_path)

    # 加载训练后的MAML模型（假设其内部包含Net1D模型）
    maml_model_path = models_src + '/metalearning.pkl'
    maml_model = load_model(maml_model_path)
    net1d_in_maml = maml_model.model.model  # 假设MAML模型结构如此

    # 加载预处理好的训练数据
    train_data = np.load(data_src + '/train_data.npy', allow_pickle=True)
    train_labels = np.load(data_src + '/train_label.npy', allow_pickle=True)

    # 选择多位患者索引（示例）
    patient_indices = [46, 78]  # 可根据需要修改

    # 存储VA段的初始和最终嵌入
    all_initial = []
    all_final = []
    valid_patient_ids = []

    for idx in patient_indices:
        patient_data = train_data[idx]
        patient_labels = train_labels[idx]
        va_mask = (patient_labels == 1)  # 筛选VA段
        va_data = patient_data[va_mask]

        if len(va_data) == 0:
            print(f"Patient {idx} has no VA segments, skipping.")
            continue

        valid_patient_ids.append(idx)
        initial_emb = extract_embeddings(net1d_model, va_data)
        final_emb = extract_embeddings(net1d_in_maml, va_data)
        all_initial.append(initial_emb)
        all_final.append(final_emb)

    # 绘制多位患者的UMAP图
    # plot_per_patient_side_by_side(all_initial, all_final, valid_patient_ids)
    # plot_combined_umap_single_fig(all_initial, all_final, valid_patient_ids)
    plot_overlapped_umap(all_initial, all_final, valid_patient_ids)

if __name__ == "__main__":
    main()