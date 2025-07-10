import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
import copy
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from MAML import MAML
from net1d import Net1D
from tran_models import ecgTransForm

# 数据和模型路径
models_src = "/data/myj/MetaVA_MAML/trained_models2-150"
# models_src = "/data/myj/MetaVA_MAML/trained_models"
# models_src = "/data/myj/MetaVA_V1/trained_models_2"
data_src = "/data/myj/MetaVA_V1/data"

# 加载模型
def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cuda'))
    print(f"Model loaded from {model_path}")
    print(f"Loaded model type: {type(model)}")
    if isinstance(model, MAML):
        return model.model
    return model

def load_model_at_epoch(model_path, epoch):
    model_path = f"{model_path}/metalearning_task_{epoch}.pkl"
    model = torch.load(model_path, map_location=torch.device('cuda'))
    print(f"Model loaded from {model_path}")
    if isinstance(model, MAML):
        return model.model
    return model

# 快速适应过程
def fast_adapt(model, support_data, support_labels, query_data_va, inner_lr=0.01, steps=5):
    """模拟 MAML 的快速适应过程"""
    learner = copy.deepcopy(model)
    optimizer = torch.optim.Adam(learner.parameters(), lr=inner_lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    # 内循环更新
    for _ in range(steps):
        learner.train()
        support_tensor = torch.tensor(support_data, dtype=torch.float32).unsqueeze(1).to('cuda')
        labels_tensor = torch.tensor(support_labels, dtype=torch.long).to('cuda')
        preds = learner(support_tensor)
        if preds is None:
            raise ValueError("Model's forward method returned None")
        loss = loss_fn(preds, labels_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 提取查询集（VA 片段）嵌入
    learner.eval()
    with torch.no_grad():
        query_tensor = torch.tensor(query_data_va, dtype=torch.float32).unsqueeze(1).to('cuda')
        embedding = learner.get_embedding(query_tensor)
    return embedding.cpu().numpy()

# 绘制多时间点的 UMAP 图
def plot_multi_checkpoint_umap(all_embeddings, checkpoints, patient_ids):
    num_checkpoints = len(checkpoints)
    fig, axes = plt.subplots(1, num_checkpoints, figsize=(6 * num_checkpoints, 5))

    for idx, checkpoint in enumerate(checkpoints):
        combined_emb = np.concatenate(all_embeddings[checkpoint], axis=0)
        umap_model = umap.UMAP(n_neighbors=7, min_dist=0.3, metric='correlation')
        umap_emb = umap_model.fit_transform(combined_emb)
        n_samples_per_patient = [len(embed) for embed in all_embeddings[checkpoint]]
        splits = np.split(umap_emb, np.cumsum(n_samples_per_patient)[:-1])
        for i, (patient_emb, pid) in enumerate(zip(splits, patient_ids)):
            axes[idx].scatter(patient_emb[:, 0], patient_emb[:, 1],
                              c=f'C{i}', alpha=0.6, marker='o', label=f'Patient {pid}')
        axes[idx].set_title(f'Checkpoint: {checkpoint}')
        axes[idx].legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("MAML_debug10.jpg")
    print("Image saved.")

# 主程序
def main():
    net1d_model = load_model(models_src + '/raw.pkl')
    maml_model = load_model(models_src + '/metalearning.pkl')
    net1d_in_maml = maml_model

    test_data = np.load(data_src + '/adapt_data.npy', allow_pickle=True)
    test_labels = np.load(data_src + '/adapt_label.npy', allow_pickle=True)
    patient_indices = range(len(test_data))

    # checkpoints = ['initial', 4, 12, 24, 60, 148, 164, 176, 196, 212, 248, 268, 284, 292, 300, 'final']
    checkpoints = ['initial', 2, 6, 14, 22, 34, 42, 62, 82, 102, 122, 134, 142, 146, 150, 'final']
    # start = 2
    # end = 160
    # step = 2
    # epochs_num = list(range(start, end, step))
    # checkpoints = ['initial'] + epochs_num + ['final']

    all_embeddings = {checkpoint: [] for checkpoint in checkpoints}
    valid_patient_ids = []

    for idx in patient_indices:
        patient_data = test_data[idx]
        patient_labels = test_labels[idx]

        # 支持集：从所有数据（标签 0 和 1）中抽取
        support_data, query_data, support_labels, query_labels = train_test_split(
            patient_data, patient_labels, test_size=0.2, random_state=42
        )

        # 查询集：从 query_data 中筛选 VA 片段（标签为 1）
        va_mask = (query_labels == 1)
        query_data_va = query_data[va_mask]

        if len(query_data_va) == 0:
            print(f"Patient {idx} has no VA segments in query set, skipping.")
            continue

        valid_patient_ids.append(idx)

        for checkpoint in checkpoints:
            if checkpoint == 'initial':
                model = net1d_model
            elif checkpoint == 'final':
                model = net1d_in_maml
            else:
                model = load_model_at_epoch(models_src, checkpoint)

            scaler = StandardScaler()
            support_data = scaler.fit_transform(support_data)
            query_data_va = scaler.transform(query_data_va)  # 对查询集应用相同的标准化
            emb = fast_adapt(model, support_data, support_labels, query_data_va)
            if emb is not None:
                all_embeddings[checkpoint].append(emb)

    plot_multi_checkpoint_umap(all_embeddings, checkpoints, valid_patient_ids)

if __name__ == "__main__":
    main()