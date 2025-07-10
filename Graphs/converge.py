import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------
# 1. 读取两种方式的收敛日志
#    - meta_file   : 预微调 + 微调 （MetaVA）
#    - base_file   : 仅微调        （Baseline）
# ------------------------------------------------------------------
meta_file = '/Users/shiqissszj/Documents/All_Projects/MetaVa/Logs/search/final_loghvpre_losses.csv'      # 修改为实际路径
base_file = '/Users/shiqissszj/Documents/All_Projects/MetaVa/Logs/search/single_losses.csv'

meta_df = pd.read_csv(meta_file)
base_df = pd.read_csv(base_file)

# 要求列名至少包含 Patient, Epoch(从1递增), Loss
assert {'Patient', 'Epoch', 'Loss'}.issubset(meta_df.columns)
assert {'Patient', 'Epoch', 'Loss'}.issubset(base_df.columns)

# ------------------------------------------------------------------
# 2. 计算每位患者收敛到 early‑stop 的 epoch 数
#    （即该患者最大 Epoch）
# ------------------------------------------------------------------
meta_epochs = meta_df.groupby('Patient')['Epoch'].max().rename('Pre-Fine-tuning')
base_epochs = base_df.groupby('Patient')['Epoch'].max().rename('Direct Finetune')

epoch_df = pd.concat([meta_epochs, base_epochs], axis=1).dropna().astype(int)
epoch_df = epoch_df.sort_index()

# ------------------------------------------------------------------
# 3. 可视化 ① 箱线 / 小提琴图
# ------------------------------------------------------------------
plt.figure(figsize=(5,3))
plot_df = epoch_df.melt(var_name='Method', value_name='Epochs')
sns.violinplot(x='Method', y='Epochs', data=plot_df,
               palette='Pastel1', inner='quartile')
sns.stripplot(x='Method', y='Epochs', data=plot_df,
              color='k', size=4, alpha=0.6, jitter=True)
plt.title('Epochs to Converge per Patient')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 4. 可视化 ② Paired slope chart  (x 轴 = Patient ID)
# ------------------------------------------------------------------
plt.figure(figsize=(8,3))                               # 宽一些
x_pos = np.arange(len(epoch_df))                        # 连续 Patient 索引
width = 0.15                                            # Baseline / Meta x 方向微移

# 依次画每位患者的连线
for i, (pid, row) in enumerate(epoch_df.iterrows()):
    plt.plot([i-width, i+width],                        # x 坐标：左右微移
             [row['Base_epoch'], row['Meta_epoch']],    # y 坐标
             color='gray', linewidth=1)

# 散点
plt.scatter(x_pos-width, epoch_df['Base_epoch'],
            color='tab:blue', label='Baseline',  s=30, zorder=3)
plt.scatter(x_pos+width, epoch_df['Meta_epoch'],
            color='tab:orange', label='MetaVA',  s=30, zorder=3)

# 轴与标签
plt.xticks(x_pos, epoch_df.index, rotation=45)          # Patient ID 作为标签
plt.ylabel('Epochs to Converge')
plt.xlabel('Patient ID')
plt.title('Per‑Patient Convergence Speed')
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 5. 可视化 ③ CDF 曲线
# ------------------------------------------------------------------
def cdf(data):
    data_sorted = np.sort(data)
    y = np.arange(1, len(data_sorted)+1) / len(data_sorted)
    return data_sorted, y

x_b, y_b = cdf(epoch_df['Base_epoch'])
x_m, y_m = cdf(epoch_df['Meta_epoch'])

plt.figure(figsize=(5,3))
plt.plot(x_b, y_b, label='Baseline')
plt.plot(x_m, y_m, label='MetaVA')
plt.xlabel('Epoch threshold')
plt.ylabel('Fraction converged')
plt.title('CDF of Epochs to Converge')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 6. 可视化 ④ 代表性患者 loss 曲线
#    选 3 个患者：min、median、max Meta  epoch
# ------------------------------------------------------------------
rep_patients = epoch_df.sort_values('Meta_epoch').index[[0,
                                                         len(epoch_df)//2,
                                                         -1]]

plt.figure(figsize=(6,3.5))
for pid in rep_patients:
    meta_trace = meta_df.loc[meta_df['Patient']==pid].sort_values('Epoch')
    base_trace = base_df.loc[base_df['Patient']==pid].sort_values('Epoch')
    plt.plot(meta_trace['Epoch'], meta_trace['Loss'],
             label=f'Patient {pid} Meta', lw=2)
    plt.plot(base_trace['Epoch'], base_trace['Loss'],
             '--', label=f'Patient {pid} Base', lw=1)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss (log)')
plt.title('Prototype Loss Curves')
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()
