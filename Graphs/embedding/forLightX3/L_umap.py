import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
from nets.nets import LightX3ECG

# 数据和模型路径（与原代码一致）
models_src = "/data/myj/Compared_Methods/LightX3ECG/trained_models"
data_src = "/data/myj/MetaVA_V1/data"

# 加载模型
def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cuda'))
    print(f"Model loaded from {model_path}")
    print(f"Loaded content type: {type(model)}")
    model.to('cuda')
    for name, module in model.named_children():
        module.to('cuda')
    model.eval()
    return model

def load_model_at_epoch(model_path, epoch):
    model_path = f"{model_path}/model_epoch_{epoch}.pkl"
    model = torch.load(model_path, map_location=torch.device('cuda'))
    print(f"Model loaded from {model_path}")
    print(f"Loaded content type: {type(model)}")
    model.to('cuda')
    model.eval()
    return model

# 提取嵌入特征
def extract_embeddings(model, query_data_va):
    model.eval()
    with torch.no_grad():
        # 假设数据是单导联，添加一个维度以匹配 (batch_size, 1, sequence_length)
        query_tensor = torch.tensor(query_data_va, dtype=torch.float32).unsqueeze(1).to('cuda')
        embedding = model.get_embedding(query_tensor).cpu().numpy()
        if np.isnan(embedding).any():
            print(f"某检查点的嵌入中发现了 NaN")
            return None
    return embedding

# 绘制多时间点的 UMAP 图
def plot_multi_checkpoint_umap(all_embeddings, checkpoints, patient_ids):
    num_checkpoints = len(checkpoints)
    fig, axes = plt.subplots(1, num_checkpoints, figsize=(6 * num_checkpoints, 5))

    for idx, checkpoint in enumerate(checkpoints):
        combined_emb = np.concatenate(all_embeddings[checkpoint], axis=0)
        if np.isnan(combined_emb).any():
            print("在 combined_emb 中检测到 NaN")
        umap_model = umap.UMAP(n_neighbors=7, min_dist=0.3, metric='correlation')
        umap_emb = umap_model.fit_transform(combined_emb)
        n_samples_per_patient = [len(embed) for embed in all_embeddings[checkpoint]]
        splits = np.split(umap_emb, np.cumsum(n_samples_per_patient)[:-1])
        for i, (patient_emb, pid) in enumerate(zip(splits, patient_ids)):
            axes[idx].scatter(patient_emb[:, 0], patient_emb[:, 1],
                              c=f'C{i}', alpha=0.6, marker='o', label=f'Patient {pid}')
        if checkpoint == 10:
            title = "Initial"
        elif checkpoint == 120:
            title = "Epoch 100"
        else:
            title = "Final"
        axes[idx].set_title(title)
        axes[idx].legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("Light3.jpg")
    print("Image saved.")

import random

def main():
    test_data = np.load(data_src + '/adapt_data.npy', allow_pickle=True)
    test_labels = np.load(data_src + '/adapt_label.npy', allow_pickle=True)
    patient_indices = range(len(test_data))

    checkpoints = [10, 120, 180]

    all_embeddings = {checkpoint: [] for checkpoint in checkpoints}
    valid_patient_ids = []

    for idx in patient_indices:
        patient_data = test_data[idx]
        patient_labels = test_labels[idx]

        # 直接使用 VA 片段作为查询集
        va_mask = (patient_labels == 1)
        query_data_va = patient_data[va_mask]

        if len(query_data_va) == 0:
            print(f"Patient {idx} has no VA segments, skipping.")
            continue

        # 随机选择 14% 的 VA 片段（与原代码保持一致）
        num_va = len(query_data_va)
        num_select = max(1, int(num_va * 0.11))  # 至少选择 1 个
        selected_indices = random.sample(range(num_va), num_select)
        query_data_va = query_data_va[selected_indices]

        valid_patient_ids.append(idx)

        for checkpoint in checkpoints:
            model = load_model_at_epoch(models_src, checkpoint)

            scaler = StandardScaler()
            query_data_va_scaled = scaler.fit_transform(query_data_va)
            if np.isnan(query_data_va_scaled).any():
                print(f"患者 {idx} 的标准化数据中发现了 NaN")

            emb = extract_embeddings(model, query_data_va_scaled)
            if emb is not None:
                all_embeddings[checkpoint].append(emb)

    plot_multi_checkpoint_umap(all_embeddings, checkpoints, valid_patient_ids)

if __name__ == "__main__":
    main()