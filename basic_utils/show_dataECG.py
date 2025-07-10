import wfdb

# # 使用 wfdb.rdsamp 读取文件
# record = wfdb.rdsamp('/Users/shiqissszj/Documents/Datasets/vtac-a/waveforms/0a42ab/0a42ab_0016')
#
# # record 是一个元组，其中第一个元素是信号数据，第二个元素是头文件信息
# signal_data, header_data = record
#
# # 打印信号数据
# print("信号数据:")
# print(signal_data)
# print(signal_data.shape)
#
# # 打印头文件信息
# print("\n头文件信息:")
# print(header_data)

ecg_data = wfdb.rdsamp('/Users/shiqissszj/Documents/Datasets/vtac/waveforms/f77db4/f77db4_0346')
# ecg_data = wfdb.rdsamp('/Users/shiqissszj/Downloads/cu28')

fs = ecg_data[1]['fs']
lead_names = ecg_data[1]['sig_name']
num_samples = ecg_data[1]['sig_len']

window_size = int(fs * 2)  # 窗口大小（采样点数）
stride = int(fs * 2)  # 滑动窗口步长（采样点数）

print(f"ECG数据：{ecg_data}")
print(f"采样率：{fs} Hz")
print(f"导联名称：{lead_names}")
print(f"信号长度：{num_samples} 个采样点")


tmp_data = ecg_data[0][:, lead_names.index('II')]
print(tmp_data)
