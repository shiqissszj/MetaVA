import torch
import wfdb
import numpy as np
import umap
import matplotlib.pyplot as plt

models_src = "/Users/shiqissszj/Documents/All_Projects/MetaVa/training_logs/Version2：MAML+CL/trained_models"
data_src = "/Users/shiqissszj/Documents/All_Projects/MetaVa/MetaVA_Codes/MetaDemo_V1/data"


# 1. 加载Net1D模型
def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    print(f"Model loaded from {model_path}")
    return model


# 2. 获取患者ECG数据
def get_patient_ecg(path, record):
    """读取ECG数据"""
    record_data = wfdb.rdsamp(path + '/' + record)  # 获取ECG信号数据
    return record_data[0]  # 返回信号数据


# 3. 从模型中提取嵌入特征
def get_embedding(model, ecg_data):
    """获取Net1D模型的嵌入特征"""
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        embedding = model.get_embedding(ecg_data)  # 获取嵌入特征
    if embedding is None:
        print("Error: Model returned None for embedding.")
    else:
        print(f"Embedding shape: {embedding.shape}")  # 调试输出，查看返回的嵌入特征形状
    return embedding


# 4. 加载Net1D模型和元学习模型
def load_maml_model(model_path):
    """加载训练后的MAML模型"""
    maml_model = torch.load(model_path, map_location=torch.device('cpu'))
    print(f"MAML model loaded from {model_path}")
    return maml_model


# 5. 准备ECG数据并提取初始和训练后的嵌入
def extract_embeddings(model, ecg_data):
    """提取嵌入特征"""
    ecg_tensor = torch.tensor(ecg_data, dtype=torch.float32)
    ecg_tensor = ecg_tensor.unsqueeze(1)  # 为每个特征添加一个通道
    print(f"ECG tensor shape: {ecg_tensor.shape}")
    embedding = get_embedding(model, ecg_tensor)
    if embedding is None:
        print("Warning: Model returned None for embedding.")
    return embedding


# 6. 使用UMAP进行降维
def plot_umap_combined(all_initial_embeddings, all_final_embeddings, valid_patient_ids):
    """合并绘制所有患者的VA段嵌入变化"""

    if not all_initial_embeddings or not all_final_embeddings:
        print("Error: No embeddings to plot.")
        return

        # 合并所有初始和最终嵌入
    combined_initial = np.concatenate(all_initial_embeddings, axis=0)
    combined_final = np.concatenate(all_final_embeddings, axis=0)

    # 使用同一个UMAP模型（用初始嵌入拟合）
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='correlation')
    umap_model.fit(combined_initial)  # 用初始嵌入拟合UMAP

    # 转换所有嵌入
    umap_initial = umap_model.transform(combined_initial)
    umap_final = umap_model.transform(combined_final)

    # 创建颜色映射（每个患者一个颜色）
    colors = plt.cm.tab10(np.linspace(0, 1, len(valid_patient_ids)))

    plt.figure(figsize=(12, 8))

    # 绘制初始嵌入（透明）和最终嵌入（不透明）
    start_idx = 0
    for i, patient_id in enumerate(valid_patient_ids):  # 使用valid_patient_ids
        # 计算当前患者的嵌入范围
        end_idx = start_idx + len(all_initial_embeddings[i])

        # 初始嵌入（半透明）
        plt.scatter(
            umap_initial[start_idx:end_idx, 0],
            umap_initial[start_idx:end_idx, 1],
            c=[colors[i]],
            alpha=0.3,
            label=f'Patient {patient_id} (Initial)',
            marker='o'
        )

        # 最终嵌入（不透明）
        plt.scatter(
            umap_final[start_idx:end_idx, 0],
            umap_final[start_idx:end_idx, 1],
            c=[colors[i]],
            alpha=0.8,
            edgecolors='k',
            label=f'Patient {patient_id} (Final)',
            marker='s'
        )

        start_idx = end_idx

    plt.title("VA Segment Embedding Trajectories")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_umap_side_by_side(all_initial_embeddings, all_final_embeddings, valid_patient_ids):
    """左右并列显示训练前后的嵌入图"""
    if not all_initial_embeddings or not all_final_embeddings:
        print("Error: No embeddings to plot.")
        return

    # 创建带两个子图的画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # 使用同一个UMAP模型（用初始嵌入拟合）
    combined_initial = np.concatenate(all_initial_embeddings, axis=0)
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='correlation')
    umap_model.fit(combined_initial)

    # 转换所有嵌入
    umap_initial = umap_model.transform(combined_initial)
    umap_final = umap_model.transform(np.concatenate(all_final_embeddings, axis=0))

    # 创建颜色映射（每个患者一个固定颜色）
    unique_patients = list(sorted(valid_patient_ids))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_patients)))
    color_map = {pid: colors[i] for i, pid in enumerate(unique_patients)}

    # 绘制初始嵌入（左侧）
    start_idx = 0
    for i, pid in enumerate(valid_patient_ids):
        end_idx = start_idx + len(all_initial_embeddings[i])
        ax1.scatter(
            umap_initial[start_idx:end_idx, 0],
            umap_initial[start_idx:end_idx, 1],
            c=[color_map[pid]],
            alpha=0.6,
            label=f'Patient {pid}',
            marker='o'
        )
        start_idx = end_idx
    ax1.set_title("Before Training")
    ax1.set_xlabel("UMAP1")
    ax1.set_ylabel("UMAP2")

    # 绘制最终嵌入（右侧）
    start_idx = 0
    for i, pid in enumerate(valid_patient_ids):
        end_idx = start_idx + len(all_final_embeddings[i])
        ax2.scatter(
            umap_final[start_idx:end_idx, 0],
            umap_final[start_idx:end_idx, 1],
            c=[color_map[pid]],
            alpha=0.6,
            edgecolors='k',  # 添加边框增强区分度
            label=f'Patient {pid}',
            marker='s'  # 使用不同标记形状
        )
        start_idx = end_idx
    ax2.set_title("After Training")
    ax2.set_xlabel("UMAP1")
    ax2.set_ylabel("UMAP2")

    # 合并图例（避免重复）
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels,
               bbox_to_anchor=(1.12, 0.5),
               loc='center right',
               title="Patient ID")

    plt.tight_layout()
    plt.show()


def plot_per_patient(initial_emb, final_emb, patient_id):
    """为单个患者绘制训练前后对比图"""
    umap_model = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation')
    combined_data = np.concatenate([initial_emb, final_emb], axis=0)
    umap_model.fit(combined_data)  # 使用该患者的所有数据训练

    # 转换数据
    umap_initial = umap_model.transform(initial_emb)
    umap_final = umap_model.transform(final_emb)

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Patient {patient_id} VA Segment Embeddings", fontsize=14)

    # 绘制初始嵌入
    ax1.scatter(umap_initial[:, 0], umap_initial[:, 1],
                c='#1f77b4', alpha=0.6, marker='o',
                edgecolors='w', linewidth=0.5)
    ax1.set_title("Before Training")
    ax1.set_xlabel("UMAP1")
    ax1.set_ylabel("UMAP2")

    # 绘制最终嵌入
    ax2.scatter(umap_final[:, 0], umap_final[:, 1],
                c='#ff7f0e', alpha=0.6, marker='s',
                edgecolors='k', linewidth=0.5)
    ax2.set_title("After Training")
    ax2.set_xlabel("UMAP1")
    ax2.set_ylabel("UMAP2")

    # 统一坐标范围
    x_min = min(umap_initial[:, 0].min(), umap_final[:, 0].min()) - 0.5
    x_max = max(umap_initial[:, 0].max(), umap_final[:, 0].max()) + 0.5
    y_min = min(umap_initial[:, 1].min(), umap_final[:, 1].min()) - 0.5
    y_max = max(umap_initial[:, 1].max(), umap_final[:, 1].max()) + 0.5

    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()


# 主程序
def main():
    # 加载初始的Net1D模型
    net1d_model_path = models_src + '/raw.pkl'  # 初始模型的路径
    net1d_model = load_model(net1d_model_path)

    # 加载训练后的元学习模型
    maml_model_path = models_src + '/metalearning.pkl'  # 元学习模型的路径
    maml_model = load_maml_model(maml_model_path)
    net1d_in_maml = maml_model.model.model  # 假设MAML模型将Net1D存储在.model属性

    # 加载预处理好的训练数据 (假设train_data.npy和train_labels.npy已存在)
    train_data = np.load(data_src + '/train_data.npy', allow_pickle=True)  # 加载训练数据
    train_labels = np.load(data_src + '/train_label.npy', allow_pickle=True)  # 加载训练标签

    print(net1d_in_maml)  # 应该显示Net1D的结构
    print(hasattr(net1d_in_maml, "get_embedding"))  # 应该返回True

    # 选择多个患者索引（示例）
    patient_indices = range(48, 91)  # 可修改为实际需要的患者索引

    # 存储所有VA段的初始和最终嵌入，以及有效的患者ID
    all_initial = []
    all_final = []
    valid_patient_ids = []  # 新增：记录实际有效的患者ID

    for idx in patient_indices:
        # 获取该患者的所有数据段
        patient_data = train_data[idx]
        patient_labels = train_labels[idx]

        # 筛选VA段（标签为1）
        va_mask = (patient_labels == 1)
        va_data = patient_data[va_mask]

        if len(va_data) == 0:
            print(f"Patient {idx} has no VA segments, skipping.")
            continue  # 跳过没有VA段的患者

        # 记录有效患者ID
        valid_patient_ids.append(idx)  # 新增

        # 提取嵌入
        initial_emb = extract_embeddings(net1d_model, va_data).cpu().numpy()
        final_emb = extract_embeddings(net1d_in_maml, va_data).cpu().numpy()

        all_initial.append(initial_emb)
        all_final.append(final_emb)

    # 绘制合并后的UMAP图（传入有效患者ID）
    # plot_umap_combined(all_initial, all_final, valid_patient_ids)
    # plot_umap_side_by_side(all_initial, all_final, valid_patient_ids)
    for i, pid in enumerate(valid_patient_ids):
        # 提取当前患者的数据
        initial_emb = all_initial[i]
        final_emb = all_final[i]

        # 绘制单个患者的对比图
        plot_per_patient(initial_emb, final_emb, pid)

if __name__ == "__main__":
    main()
