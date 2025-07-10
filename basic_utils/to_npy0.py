import os
import wfdb
import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 用户需根据自己的目录路径修改
PTBXL_PATH = '/Users/shiqissszj/Documents/Datasets/PTB-XL'  # PTB-XL数据集根目录，包含ptbxl_database.csv和records500等文件夹
OUTPUT_NPY_DIR = './ptbxl_npy/'  # 存放单独npy文件的目录
TSV_DIR = './ptbxl_tsv/'  # 存放train.tsv、valid.tsv、test.tsv的目录
SAMPLING_RATE = 500  # 使用500Hz的数据，即filename_hr

os.makedirs(OUTPUT_NPY_DIR, exist_ok=True)
os.makedirs(TSV_DIR, exist_ok=True)

# 1. 加载数据集注释信息
Y = pd.read_csv(os.path.join(PTBXL_PATH, 'ptbxl_database.csv'), index_col='ecg_id')
# 解析scp_codes
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# 根据采样率确定使用filename_lr或filename_hr
if SAMPLING_RATE == 100:
    filenames = Y.filename_lr
elif SAMPLING_RATE == 500:
    filenames = Y.filename_hr
else:
    raise ValueError("Unsupported sampling rate")

# 2. 从scp_codes中提取标签逻辑（此处仅演示，将有无scp_codes简单映射为0或1）
# 实际使用中请根据PTB-XL官方诊断分类方案从scp_codes提取对应标签。
def extract_label(scp_codes_dict):
    # 简单例子：如果scp_codes不为空则label=1，否则=0
    return 1 if len(scp_codes_dict) > 0 else 0

Y['label'] = Y.scp_codes.apply(extract_label)

# 3. 遍历所有ECG记录，将每条记录单独保存为npy文件
data_list = []
for ecg_id in Y.index:
    ecg_label = Y.loc[ecg_id, 'label']
    dat_path = os.path.join(PTBXL_PATH, Y.loc[ecg_id, 'filename_hr'])  # 对500Hz数据使用filename_hr
    record = wfdb.rdrecord(dat_path)  # 去掉后缀.dat
    signals = record.p_signal  # shape: [samples, leads]

    # 将该ECG记录保存为npy文件
    npy_filename = f"ecg_{ecg_id}.npy"
    npy_path = os.path.join(OUTPUT_NPY_DIR, npy_filename)
    np.save(npy_path, signals)
    # data_list中存放 (npy_path, label)
    data_list.append((npy_path, ecg_label))

# 4. 划分训练/验证/测试集
train_ratio = 0.7
valid_ratio = 0.15
test_ratio = 0.15

train_val, test = train_test_split(data_list, test_size=test_ratio, random_state=42, shuffle=True)
train, valid = train_test_split(train_val, test_size=valid_ratio/(train_ratio+valid_ratio), random_state=42, shuffle=True)

# 5. 保存tsv文件
def save_tsv(data, tsv_path):
    df = pd.DataFrame(data, columns=['file_path', 'label'])
    df.to_csv(tsv_path, sep='\t', index=False)

save_tsv(train, os.path.join(TSV_DIR, 'train.tsv'))
save_tsv(valid, os.path.join(TSV_DIR, 'valid.tsv'))
save_tsv(test, os.path.join(TSV_DIR, 'test.tsv'))

print("数据准备完成：")
print(f"训练集: {len(train)} 条记录")
print(f"验证集: {len(valid)} 条记录")
print(f"测试集: {len(test)} 条记录")
print(f"独立npy文件存放目录: {OUTPUT_NPY_DIR}")
print(f"TSV文件存放目录: {TSV_DIR}")