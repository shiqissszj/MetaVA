import wfdb
import numpy as np

# 读取 .atr 文件
annotation = wfdb.rdann('/Users/shiqissszj/Documents/Datasets/mitdb/233', 'atr')  # 替换 'example_record' 为你的实际文件名
ecg_data = wfdb.rdsamp('/Users/shiqissszj/Documents/Datasets/mitdb/233')

fs = ecg_data[1]['fs']
lead_names = ecg_data[1]['sig_name']
num_samples = ecg_data[1]['sig_len']

print(f"ECG数据：{ecg_data}")
print(f"采样率：{fs} Hz")
print(f"导联名称：{lead_names}")
print(f"信号长度：{num_samples} 个采样点")

# 输出属性内容
print("采样点 (sample):", annotation.sample)
print("符号 (symbol):", annotation.symbol)
print("辅助信息 (aux_note):", annotation.aux_note)

# 将属性转换为 NumPy 数组
samples = np.array(annotation.sample)  # 转换采样点为 NumPy 数组
symbols = np.array(annotation.symbol)  # 转换符号为 NumPy 数组
aux_notes = np.array(annotation.aux_note)  # 转换辅助信息为 NumPy 数组

print(samples.shape)
print(symbols.shape)
print(aux_notes.shape)
