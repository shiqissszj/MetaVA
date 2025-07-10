# import wfdb
#
# # 读取注释
# annotation = wfdb.rdann('/Users/shiqissszj/Downloads/cu06', 'atr')
#
# # print(annotation['sample'].shape)
#
# # 检查属性
# print("sample 属性:", hasattr(annotation, 'sample'))  # True
# print("symbol 属性:", hasattr(annotation, 'symbol'))  # True
# print("aux_note 属性:", hasattr(annotation, 'aux_note'))  # True
#
# # 输出属性内容
# print("采样点 (sample):", annotation.sample.shape)  # 打印前10个采样点
# print("符号 (symbol):", annotation.symbol.shape)    # 打印前10个符号
# print("辅助信息 (aux_note):", annotation.aux_note)  # 打印前10个辅助信息

import wfdb
import numpy as np

# 读取 .atr 文件
annotation = wfdb.rdann('/Users/shiqissszj/Documents/Datasets/vfdb/418', 'atr')  # 替换 'example_record' 为你的实际文件名
ecg_data = wfdb.rdsamp('/Users/shiqissszj/Documents/Datasets/vfdb/418')

fs = ecg_data[1]['fs']
lead_names = ecg_data[1]['sig_name']
num_samples = ecg_data[1]['sig_len']

window_size = int(fs * 2)  # 窗口大小（采样点数）
stride = int(fs * 2)  # 滑动窗口步长（采样点数）

print(f"ECG数据：{ecg_data}")
print(f"采样率：{fs} Hz")
print(f"导联名称：{lead_names}")
print(f"信号长度：{num_samples} 个采样点")

# 输出属性内容
print("采样点 (sample):", annotation.sample)
print("符号 (symbol):", annotation.symbol)

# 将属性转换为 NumPy 数组
samples = np.array(annotation.sample)  # 转换采样点为 NumPy 数组
symbols = np.array(annotation.symbol)  # 转换符号为 NumPy 数组
aux_notes = np.array(annotation.aux_note)  # 转换辅助信息为 NumPy 数组
aux_notes0 = np.array([i.strip('\x00') for i in aux_notes])
print("辅助信息 (aux_note):", aux_notes0)

print(samples.shape)
print(symbols.shape)
print(aux_notes.shape)


# # 检查形状
# print("采样点 (sample) 的形状:", samples.shape)
# print("符号 (symbol) 的形状:", symbols.shape)
# print("辅助信息 (aux_note) 的形状:", aux_notes.shape)
#
# # 验证是否匹配
# if samples.shape == symbols.shape == aux_notes.shape:
#     print("采样点、符号和辅助信息的形状匹配！")
# else:
#     print("采样点、符号和辅助信息的形状不匹配！")