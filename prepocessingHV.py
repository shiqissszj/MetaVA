import os, json, glob, csv, math, numpy as np
from collections import defaultdict
from tqdm import tqdm
from scipy.signal import resample_poly

# --------- 路径自行修改 --------- #
VT_DIR = "/Users/shiqissszj/Documents/All_Projects/MetaVa/data/VT"
VTN_DIR = "/Users/shiqissszj/Documents/All_Projects/MetaVa/data/VT_N"
DST_DIR = "./dataHV/"
# -------------------------------- #

WIN_SEC, FS_TARGET = 2.0, 200
WIN_SAMPLES = int(WIN_SEC * FS_TARGET)  # 400


def resample(sig, fs_orig):
    if fs_orig == FS_TARGET: return sig.astype(np.float32)
    g = math.gcd(fs_orig, FS_TARGET)
    return resample_poly(sig, FS_TARGET // g, fs_orig // g).astype(np.float32)


def segment(sig, step):
    idx = np.arange(0, len(sig) - WIN_SAMPLES + 1, step, dtype=int)
    return np.stack([sig[i:i + WIN_SAMPLES] for i in idx])


def parse_ecg(o):
    return np.asarray(o, np.float32) if isinstance(o, list) \
        else np.fromstring(o.strip("{}"), sep=",", dtype=np.float32)


def collect():
    ud = defaultdict(list)

    # VT 正样本
    for jf in tqdm(glob.glob(os.path.join(VT_DIR, "*.json")), desc="VT"):
        j = json.load(open(jf, encoding="utf-8"))
        uid = str(j.get("userid") or j.get("user_id") or j["user"])
        ud[uid].append((parse_ecg(j["ecgdata"]), 1, int(j["sample_rate"])))

    # VT_N 负样本
    for jf in tqdm(glob.glob(os.path.join(VTN_DIR, "*.json")), desc="VT_N"):
        j = json.load(open(jf, encoding="utf-8"))
        if "RECORDS" in j:
            for rec in j["RECORDS"]:
                ud[str(rec["user_id"])].append(
                    (parse_ecg(rec["ecg_data"]), 0, int(rec["sample_rate"])))
        else:
            uid = str(j.get("userid") or j["user_id"])
            ud[uid].append((parse_ecg(j["ecgdata"]), 0, int(j["sample_rate"])))
    return ud


def main():
    users = collect()
    data_lst, label_lst = [], []
    meta = [["user_id", "n_seg", "n_pos", "n_neg"]]

    for uid, segs in tqdm(users.items(), desc="Users"):
        seg_all, lab_all = [], []
        for sig_raw, lab, fs in segs:
            sig_200 = resample(sig_raw, fs)
            step = 60 if lab == 1 else WIN_SAMPLES  # 1 s or 2 s
            wins = segment(sig_200, step)
            seg_all.append(wins)
            lab_all.append(np.full(len(wins), lab, np.int64))
        seg_all = np.concatenate(seg_all);
        lab_all = np.concatenate(lab_all)
        data_lst.append(seg_all);
        label_lst.append(lab_all)
        meta.append([uid, len(lab_all),
                     int((lab_all == 1).sum()), int((lab_all == 0).sum())])

    os.makedirs(DST_DIR, exist_ok=True)
    np.save(os.path.join(DST_DIR, "adapt_data.npy"),
            np.array(data_lst, dtype=object))
    np.save(os.path.join(DST_DIR, "adapt_label.npy"),
            np.array(label_lst, dtype=object))
    with open(os.path.join(DST_DIR, "meta.csv"), "w", newline="") as f:
        csv.writer(f).writerows(meta)

    print(f"✓ 完成：{len(data_lst)} 位患者，共 {sum(len(x) for x in data_lst)} 段")


if __name__ == "__main__":
    main()
