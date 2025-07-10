# convert_npz_folder.py
import numpy as np
import pandas as pd
from pathlib import Path

# ------------ 请按需修改 ------------
SRC_DIR = Path("/Users/shiqissszj/Documents/All_Projects/MetaVa/VT_others")
DST_DIR = SRC_DIR.parent / "VT_others_clean"
DST_DIR.mkdir(exist_ok=True)
CSV_PATH = DST_DIR / "metadata.csv"
# ------------------------------------

rows = []

def parse_ecg_string(s: str) -> np.ndarray:
    """把形如 '{0.1,0.2,...}' 的字符串转成 float32 numpy 一维数组"""
    s = s.strip("{} \n")
    return np.fromstring(s, sep=",", dtype=np.float32)

for src_path in SRC_DIR.glob("*.npz"):
    with np.load(src_path, allow_pickle=False) as f:
        rec_id   = f["recordid"].item()
        label    = f["labels"].item()
        fs       = int(f["sample_rate"].item())
        ecg_str  = f["ecgdata"].item()
        uploader = f["reply_user_name"].item()

    ecg_arr = parse_ecg_string(ecg_str)
    n_pts   = ecg_arr.size
    duration_sec = n_pts / fs if fs else 0.0

    # ---- 保存干净版 npz ---- #
    dst_path = DST_DIR / (src_path.stem + "_clean.npz")
    np.savez_compressed(
        dst_path,
        signals=ecg_arr,      # (N,)
        sample_rate=fs,       # 125
        label=label,          # 'VT'
        recordid=rec_id,
        uploader=uploader,
    )

    # ---- 把元信息收集到表格 ---- #
    rows.append(
        dict(
            filename=dst_path.name,
            recordid=rec_id,
            label=label,
            sample_rate=fs,
            n_points=n_pts,
            duration_sec=round(duration_sec, 2),
            uploader=uploader,
        )
    )

# ---- 写出 CSV ---- #
df = pd.DataFrame(rows)
df = df.sort_values("recordid")        # 可按需要排序
df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
print(f"✅ 完成！清洗文件存于: {DST_DIR}")
print(f"✅ 汇总信息已写入: {CSV_PATH}")
