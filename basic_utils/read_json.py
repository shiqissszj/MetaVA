import json, numpy as np, pandas as pd
from pathlib import Path

file_paths = [
    Path("/Users/shiqissszj/Documents/All_Projects/MetaVa/data/VT/417018932558303232.json"),
    Path("/Users/shiqissszj/Documents/All_Projects/MetaVa/data/VT_N/417018931132239872.json")
]

summary_rows = []
keys_info = {}

def parse_signal(s):
    return np.fromstring(s.strip("{} \n"), sep=",", dtype=np.float32)

for fp in file_paths:
    with fp.open() as f:
        j = json.load(f)

    if "ecgdata" in j:  # 单段
        keys_info[fp.name] = list(j.keys())
        sig = np.array(j["ecgdata"], dtype=np.float32)
        fs  = int(j["sample_rate"])
        summary_rows.append({
            "source_file": fp.name,
            "recordid": j.get("recordid", ""),
            "label": j.get("labels", ""),
            "segments": 1,
            "sample_rate(Hz)": fs,
            "points": len(sig),
            "duration(s)": round(len(sig) / fs, 2)
        })
    else:               # 多段
        records = j["RECORDS"]
        keys_info[fp.name] = list(records[0].keys())
        for rec in records:
            sig = parse_signal(rec["ecg_data"])
            fs  = int(rec["sample_rate"])
            summary_rows.append({
                "source_file": fp.name,
                "recordid": rec.get("recordid", ""),
                "label": rec.get("diaglabel", ""),
                "segments": len(records),
                "sample_rate(Hz)": fs,
                "points": len(sig),
                "duration(s)": round(len(sig) / fs, 2)
            })

df = pd.DataFrame(summary_rows)

print(df)

# —— 2) 可选：写 CSV 方便用表格软件看 —— #
df.to_csv("ECG_summary.csv", index=False, encoding="utf-8-sig")
print("已输出 CSV 至 ECG_summary.csv")
