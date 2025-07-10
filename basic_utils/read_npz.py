import numpy as np

npz_path = "/Users/shiqissszj/Documents/All_Projects/MetaVa/VT_others/VT_771489599590502400.npz"

with np.load(npz_path, allow_pickle=False) as f:
    print("=== 文件包含的所有键 ===")
    for k in f.files:
        arr = f[k]
        # 打印 shape / dtype；若是一维标量就直接打印内容
        if arr.shape == () or arr.shape == ():
            print(f"{k:15s} -> 标量, dtype={arr.dtype}, value={arr}")
        else:
            # 只看前 3 个元素避免刷屏
            preview = arr.flatten()[:3]
            print(f"{k:15s} -> shape={arr.shape}, dtype={arr.dtype}, first3={preview}")

    # ------ 专门检查 ECG 数据与标签 ------ #
    ecg      = f["ecgdata"]          # 波形
    fs       = int(f["sample_rate"]) # 采样率（Hz）
    labels   = f["labels"]

print("\n=== ECG 详细信息 ===")
print("ecg shape      :", ecg.shape)          # 例如 (L,) 或 (leads, L) 或 (segments, leads, L)
print("dtype          :", ecg.dtype)
# 支持 12 导联或多段可再拆分；这里默认把最后一维看成时间长度
length_points = ecg.shape[-1] if ecg.ndim >= 1 else 0
print("length (pts)   :", length_points)
print("sample rate    :", fs, "Hz")
print("duration       :", length_points / fs, "seconds")

print("\n=== Label 信息 ===")
print("labels shape   :", labels.shape)
print("labels dtype   :", labels.dtype)
print("labels content :", labels)             # 若太长可只打印前几个

# with np.load(npz_path, allow_pickle=False) as f:
#     txt = f["ecgdata"].item()  # 转成 Python 字符串
#     label = f["labels"].item()  # 'VT'
#     fs = int(f["sample_rate"].item())  # 125
#
# # 1. 去掉花括号与空白，按逗号分割
# txt = txt.strip("{} \n")
# ecg = np.fromstring(txt, sep=",", dtype=np.float32)  # (N,)
#
# print("点数 :", ecg.size)
# print("时长 :", ecg.size / fs, "秒")
# print("标签 :", label)
