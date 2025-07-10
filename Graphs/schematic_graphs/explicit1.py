import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.interpolate import splrep, splev
import os

# 符号到类别的映射字典，用于识别VA区域
m = {
    'N': 'SN', 'L': 'LBBB', 'R': 'RBBB', 'B': 'IVB', 'A': 'PAC', 'a': 'PAC',
    'J': 'PJC', 'S': 'PSC', 'V': 'PVC', 'r': 'PVC', 'F': 'PVC', 'e': 'AE',
    'j': 'JE', 'n': 'SE', 'E': 'VE', '/': 'PACED', 'f': 'PACED', 'Q': 'OTHER',
    '?': 'OTHER', '[': 'VF', '!': 'VF', ']': 'VF', 'x': 'PAC', '(AB': 'PAC',
    '(AFIB': 'AF', '(AF': 'AF', '(AFL': 'AFL', '(ASYS': 'PAUSE', '(B': 'PVC',
    '(BI': 'AVBI', '(BII': 'AVBII', '(HGEA': 'PVC', '(IVR': 'VE', '(N': 'SN',
    '(NOD': 'JE', '(P': 'PACED', '(PM': 'PACED', '(PREX': 'WPW', '(SBR': 'SNB',
    '(SVTA': 'SVT', '(T': 'PVC', '(VER': 'VE', '(VF': 'VF', '(VFL': 'VFL', '(VT': 'VT'
}

def butter_lowpass(cutoff, fs, order=5):
    """设计低通滤波器"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    """应用低通滤波器去噪"""
    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)

def moving_average(data, window_size=5):
    """应用移动平均平滑"""
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def process_annotations(annotation, signal_length):
    """处理注释信息，生成与信号等长的辅助注释列表"""
    full_aux_list = [''] * signal_length
    for i in range(len(annotation.symbol)):
        idx = annotation.sample[i]
        symbol = annotation.symbol[i]
        aux_note = annotation.aux_note[i].strip('\x00') if i < len(annotation.aux_note) else ''

        if symbol in ['[', '!']:
            full_aux_list[idx] = '(VF'
        elif symbol == ']':
            full_aux_list[idx] = '(N'
        else:
            full_aux_list[idx] = aux_note

    current_aux = ''
    for i in range(signal_length):
        if full_aux_list[i]:
            current_aux = full_aux_list[i]
        else:
            full_aux_list[i] = current_aux
    return full_aux_list

def get_vfvt_windows(record_path, fs, signal_length):
    """获取VF/VT时间窗口（VA区域）"""
    annotation = wfdb.rdann(record_path, 'atr')
    full_aux_list = process_annotations(annotation, signal_length)

    window_size = int(0.01 * fs)  # 窗口大小为0.01秒
    stride = window_size

    vfvt_windows = []
    start = 0
    while start + window_size <= signal_length:
        end = start + window_size
        symbols = []
        for beat_sample, beat_sym in zip(annotation.sample, annotation.symbol):
            if start <= beat_sample < end:
                symbols.append(beat_sym)
        symbols += full_aux_list[start:end]

        vfvt_flag = False
        for sym in set(symbols):
            if sym in m and m[sym] in ['VF', 'VFL', 'VT']:
                vfvt_flag = True
                break

        if vfvt_flag:
            vfvt_windows.append((start / fs, end / fs))
        start += stride

    return vfvt_windows

def plot_ecg_with_vfvt(record_path, start_time=0, end_time=None, lead_name='ECG'):
    """绘制ECG波形并在VA区域添加动态宽度的红色阴影"""
    # 读取信号
    signals, fields = wfdb.rdsamp(record_path)
    fs = fields['fs']
    signal_length = len(signals)

    # 选择导联
    try:
        lead_idx = fields['sig_name'].index(lead_name)
    except ValueError:
        print(f"未找到导联 {lead_name}，使用第一个导联")
        lead_idx = 0

    ecg = signals[:, lead_idx]
    ecg = np.nan_to_num(ecg)  # 处理缺失值

    # 加强去噪：降低截止频率到30Hz
    ecg = lowpass_filter(ecg, cutoff=30, fs=fs)

    # 插值平滑波形：20倍插值
    time = np.arange(len(ecg)) / fs
    tck = splrep(time, ecg, s=0)
    time_interp = np.linspace(time[0], time[-1], len(time) * 20)
    ecg_interp = splev(time_interp, tck, der=0)

    # 应用5点移动平均进一步平滑
    ecg_interp = moving_average(ecg_interp, window_size=5)

    # 确定时间区间
    if end_time is None:
        end_time = signal_length / fs
    mask = (time_interp >= start_time) & (time_interp <= end_time)
    time_segment = time_interp[mask]
    ecg_segment = ecg_interp[mask]

    # 获取VA时间段
    vfvt_windows = get_vfvt_windows(record_path, fs, signal_length)
    vfvt_windows_segment = [(start, end) for start, end in vfvt_windows
                            if start >= start_time and end <= end_time]

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(time_segment, ecg_segment, linewidth=1.2, color='#1f77b4', alpha=0.8)

    # 在VA区域添加动态宽度的红色阴影
    for start, end in vfvt_windows_segment:
        mask_va = (time_segment >= start) & (time_segment <= end)
        time_va = time_segment[mask_va]
        ecg_va = ecg_segment[mask_va]
        if len(ecg_va) > 0:
            # 计算每个点的局部标准差
            window_size = int(0.01 * fs * 20)  # 0.01秒窗口，考虑20倍插值
            half_window = window_size // 2
            local_std = np.zeros_like(ecg_va)
            for i in range(len(ecg_va)):
                start_idx = max(0, i - half_window)
                end_idx = min(len(ecg_va), i + half_window + 1)
                local_std[i] = np.std(ecg_va[start_idx:end_idx])
            # 计算动态宽度数组
            width = np.maximum(0.05, local_std * 0.5)  # 最小宽度0.05mV
            upper = ecg_va + width
            lower = ecg_va - width
            plt.fill_between(time_va, lower, upper, color='red', alpha=0.2)

    plt.title(f'ECG - {os.path.basename(record_path)} ({lead_name} Lead, {start_time}-{end_time}s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (mV)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

# 使用示例
if __name__ == "__main__":
    mitdb_path = "/Users/shiqissszj/Documents/Datasets/mitdb"
    record_name = "106"
    start_time = 176
    end_time = 179.5
    plot_ecg_with_vfvt(os.path.join(mitdb_path, record_name), start_time=start_time, end_time=end_time)