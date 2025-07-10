# import wfdb
#
# # 读取 .hea 文件
# record = wfdb.rdheader('/Users/shiqissszj/Documents/Datasets/vtac-a/waveforms/0a7cd2/0a7cd2_0071')  # 替换为你的文件名
#
# # 获取采样率和数据点数
# sampling_rate = record.fs
# num_samples = record.sig_len
#
# # 计算时长
# duration = num_samples / sampling_rate
# print(f"ECG 记录时长: {duration} 秒")

import os
import wfdb


def calculate_ecg_duration(folder_path):
    """
    遍历指定文件夹下的所有 .dat 文件，计算并展示每个文件的 ECG 时长。
    :param folder_path: 包含 .dat 和 .hea 文件的文件夹路径
    """
    # 遍历文件夹，找到所有的 .dat 文件
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.dat'):
                # 获取对应的 .hea 文件路径
                record_name = os.path.splitext(file)[0]  # 去掉扩展名
                hea_path = os.path.join(root, record_name)

                try:
                    # 读取 .hea 文件
                    record = wfdb.rdheader(hea_path)

                    # 获取采样率和数据点数
                    sampling_rate = record.fs
                    num_samples = record.sig_len

                    # 计算时长
                    duration = num_samples / sampling_rate
                    print(f"记录名: {record_name}, ECG 时长: {duration:.2f} 秒")
                except Exception as e:
                    print(f"无法读取记录 {record_name}: {e}")


# 设置要处理的文件夹路径
folder_path = '/Users/shiqissszj/Documents/Datasets/vtac-a/waveforms'

# 调用函数计算时长
calculate_ecg_duration(folder_path)
