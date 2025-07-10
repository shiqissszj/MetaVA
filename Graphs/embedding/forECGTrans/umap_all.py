import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
from models import ecgTransForm

# 数据和模型路径
models_src = "/data/myj/Compared_Methods/ECGTransForm/experiments_logs/exp1/binary_ecg_14_40"
data_src = "/data/myj/MetaVA_V1/data"


# 加载模型
def load_model(model_path, configs, hparams):
    content = torch.load(model_path, map_location=torch.device('cuda'))
    print(f"Model loaded from {model_path}")
    print(f"Loaded content type: {type(content)}")

    model = ecgTransForm(configs, hparams)
    if isinstance(content, dict):  # 如果是 state_dict
        model.load_state_dict(content)
    else:  # 如果是完整模型对象
        model = content
    model.to('cuda')
    model.eval()
    return model


def load_model_at_epoch(model_path, epoch, configs, hparams):
    model_path = f"{model_path}/model_epoch_{epoch}.pkl"
    content = torch.load(model_path, map_location=torch.device('cuda'))
    print(f"Model loaded from {model_path}")
    print(f"Loaded content type: {type(content)}")

    model = ecgTransForm(configs, hparams)
    if isinstance(content, dict):  # 如果是 state_dict
        model.load_state_dict(content)
    else:  # 如果是完整模型对象
        model = content
    model.to('cuda')
    model.eval()
    return model


# 提取嵌入特征
def extract_embeddings(model, query_data_va):
    model.eval()
    with torch.no_grad():
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
        umap_model = umap.UMAP(n_neighbors=9, min_dist=0.3, metric='correlation')
        umap_emb = umap_model.fit_transform(combined_emb)
        n_samples_per_patient = [len(embed) for embed in all_embeddings[checkpoint]]
        splits = np.split(umap_emb, np.cumsum(n_samples_per_patient)[:-1])
        for i, (patient_emb, pid) in enumerate(zip(splits, patient_ids)):
            axes[idx].scatter(patient_emb[:, 0], patient_emb[:, 1],
                              c=f'C{i}', alpha=0.6, marker='o', label=f'Patient {pid}')
        if checkpoint == 15:
            title = "Initial"
        elif checkpoint == 75:
            title = "Epoch 100"
        else:
            title = "Final"
        axes[idx].set_title(title)
        axes[idx].legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("ecgTransForm4.jpg")
    print("Image saved.")


import random


def main():
    # 定义 configs 和 hparams
    configs = {
        "input_channels": 1,
        "mid_channels": 32,
        "final_out_channels": 128,
        "stride": 1,
        "dropout": 0.2,
        "trans_dim": 128,
        "num_heads": 8
    }
    hparams = {"feature_dim": 128}

    test_data = np.load(data_src + '/adapt_data.npy', allow_pickle=True)
    test_labels = np.load(data_src + '/adapt_label.npy', allow_pickle=True)
    patient_indices = range(len(test_data))

    checkpoints = [15, 75, 180]

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

        num_va = len(query_data_va)
        num_select = max(1, int(num_va * 0.1))  # 至少选择 1 个
        selected_indices = random.sample(range(num_va), num_select)
        query_data_va = query_data_va[selected_indices]

        valid_patient_ids.append(idx)

        for checkpoint in checkpoints:
            model = load_model_at_epoch(models_src, checkpoint, configs, hparams)

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
