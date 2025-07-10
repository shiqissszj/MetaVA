import wfdb
import os

# 符号到VA类别的映射字典，用于识别VA事件
m = {
    'V': 'PVC',  # 室性期前收缩
    'F': 'PVC',  # 室性期前收缩（融合心搏）
    'E': 'VE',  # 室性逸搏
    '[': 'VF',  # 心室颤动开始
    '!': 'VF',  # 心室颤动持续
    ']': 'VF',  # 心室颤动结束
    '(VT': 'VT',  # 室性心动过速
    '(VFL': 'VFL'  # 心室扑动
}


def get_va_events(record_path, start_time, end_time):
    """获取指定时间区间内的VA事件"""
    # 加载注释文件
    annotation = wfdb.rdann(record_path, 'atr')

    # 获取采样频率
    record, fields = wfdb.rdsamp(record_path)
    fs = fields['fs']  # Correctly access fs from fields dictionary

    # 将时间转换为样本点
    start_sample = int(start_time * fs)
    end_sample = int(end_time * fs)

    # 获取在时间区间内的心搏和节律注释
    va_events = []
    for i, (sample, symbol) in enumerate(zip(annotation.sample, annotation.symbol)):
        if start_sample <= sample < end_sample:
            if symbol in m:
                va_events.append((sample / fs, m[symbol]))
            elif i < len(annotation.aux_note) and annotation.aux_note[i].strip():
                aux = annotation.aux_note[i].strip()
                if aux in m:
                    va_events.append((sample / fs, m[aux]))

    return va_events


def analyze_patients(record_paths, start_times, end_times):
    """分析四位患者的VA事件并输出结果"""
    for record_path, start_time, end_time in zip(record_paths, start_times, end_times):
        patient_id = os.path.basename(record_path)
        print(f"Patient {patient_id} (Time: {start_time}s - {end_time}s):")
        va_events = get_va_events(record_path, start_time, end_time)
        if va_events:
            for time, event in va_events:
                print(f" - {event} at {time:.2f}s")
        else:
            print(" - No VA events detected in this interval.")
        print()


if __name__ == "__main__":
    # 设置MIT-BIH数据库路径
    mitdb_path = "/Users/shiqissszj/Documents/Datasets/mitdb"  # 请替换为您的MIT-BIH数据库路径

    # 四位患者的记录路径和时间区间
    record_paths = [
        os.path.join(mitdb_path, "106"),
        os.path.join(mitdb_path, "203"),
        os.path.join(mitdb_path, "205"),
        os.path.join(mitdb_path, "221")
    ]
    start_times = [176, 268.5, 1476, 784]
    end_times = [179, 271, 1478.5, 787]

    # 分析并输出VA事件
    analyze_patients(record_paths, start_times, end_times)