import os
import wfdb


def calculate_ecg_leads(folder_path):
    """
    遍历指定文件夹下的所有 .dat 文件，计算并展示每个文件的 ECG 使用的导联名。
    :param folder_path: 包含多个子文件夹的根目录路径，每个子文件夹中包含 .dat 和 .hea 文件。
    """
    # 遍历文件夹和子文件夹，找到所有的 .dat 文件
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.dat'):  # 筛选出 .dat 文件
                # 获取对应的记录名（去掉扩展名）
                record_name = os.path.splitext(file)[0]  # 去掉 .dat 扩展名
                record_path = os.path.join(root, record_name)  # 构造记录文件路径，不带扩展名

                try:
                    # 使用 wfdb 读取 .hea 文件中的信号信息
                    record = wfdb.rdsamp(record_path)

                    # 提取导联信息
                    sig_name = record[1]['sig_name']

                    # 打印记录名和导联信息
                    print(f"记录名: {record_name}, ECG使用导联名: {sig_name}")
                except Exception as e:
                    # 如果读取失败，打印错误信息
                    print(f"无法读取记录 {record_name}: {e}")


# 设置要处理的根目录路径
folder_path = '/Users/shiqissszj/Documents/Datasets/vtac-a/waveforms'

# 调用函数遍历并展示导联信息
calculate_ecg_leads(folder_path)