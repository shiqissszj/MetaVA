import numpy as np
import wfdb
import os

m = {'N': 'SN',  # Normal beat (displayed as "·" by the PhysioBank ATM, LightWAVE, pschart, and psfd)
     'L': 'LBBB',  # Left bundle branch block beat
     'R': 'RBBB',  # Right bundle branch block beat
     'B': 'IVB',  # Bundle branch block beat (unspecified)
     'A': 'PAC',  # Atrial premature beat
     'a': 'PAC',  # Aberrated atrial premature beat
     'J': 'PJC',  # Nodal (junctional) premature beat
     'S': 'PSC',  # Supraventricular premature or ectopic beat (atrial or nodal)
     'V': 'PVC',  # Premature ventricular contraction
     'r': 'PVC',  # R-on-T premature ventricular contraction
     'F': 'PVC',  # Fusion of ventricular and normal beat
     'e': 'AE',  # Atrial escape beat
     'j': 'JE',  # Nodal (junctional) escape beat
     'n': 'SE',  # Supraventricular escape beat (atrial or nodal)
     'E': 'VE',  # Ventricular escape beat
     '/': 'PACED',  # Paced beat
     'f': 'PACED',  # Fusion of paced and normal beat
     'Q': 'OTHER',  # Unclassifiable beat
     '?': 'OTHER',  # Beat not classified during learning
     '[': 'VF',  # Start of ventricular flutter/fibrillation
     '!': 'VF',  # Ventricular flutter wave
     ']': 'VF',  # End of ventricular flutter/fibrillation
     'x': 'PAC',  # Non-conducted P-wave (blocked APC)
     '(AB': 'PAC',  # Atrial bigeminy
     '(AFIB': 'AF',  # Atrial fibrillation
     '(AF': 'AF',  # Atrial fibrillation
     '(AFL': 'AFL',  # Atrial flutter
     '(ASYS': 'PAUSE',  # asystole
     '(B': 'PVC',  # Ventricular bigeminy
     '(BI': 'AVBI',  # 1° heart block
     '(BII': 'AVBII',  # 2° heart block
     '(HGEA': 'PVC',  # high grade ventricular ectopic activity
     '(IVR': 'VE',  # Idioventricular rhythm
     '(N': 'SN',  # Normal sinus rhythm
     '(NOD': 'JE',  # Nodal (A-V junctional) rhythm
     '(P': 'PACED',  # Paced rhythm
     '(PM': 'PACED',  # Paced rhythm
     '(PREX': 'WPW',  # Pre-excitation (WPW)
     '(SBR': 'SNB',  # Sinus bradycardia
     '(SVTA': 'SVT',  # Supraventricular tachyarrhythmia
     '(T': 'PVC',  # Ventricular trigeminy
     '(VER': 'VE',  # ventricular escape rhythm
     '(VF': 'VF',  # Ventricular fibrillation
     '(VFL': 'VFL',  # Ventricular flutter
     '(VT': 'VT'  # ≈
     }

labels = list(set(m.values()))


def get_label_map(labels):
    out_labels = []
    for i in labels:
        if i in m:
            if m[i] in ['VF', 'VFL', 'VT']:
                out_labels.append('VF/VT')
            else:
                out_labels.append('Others')
    out_labels = list(np.unique(out_labels))
    return out_labels


def find_vfvt_records(path):
    valid_lead = ['MLII', 'ECG']  # 提取有效的导联
    vfvt_records = []  # 用来存储包含VF/VT的记录名

    with open(os.path.join(path, 'RECORDS'), 'r') as fin:
        all_record_name = fin.read().strip().split('\n')

    for record_name in all_record_name:
        try:
            tmp_ann_res = wfdb.rdann(path + '/' + record_name, 'atr').__dict__
            tmp_data_res = wfdb.rdsamp(path + '/' + record_name)
        except:
            print(f'Reading data failed for {record_name}')
            continue

        fs = tmp_data_res[1]['fs']
        window_size = int(fs * 2)  # 设置窗口大小为2秒
        stride = int(fs * 2)  # 步幅为2秒

        lead_in_data = tmp_data_res[1]['sig_name']

        # 只处理有效的导联
        my_lead_all = [tmp_lead for tmp_lead in valid_lead if tmp_lead in lead_in_data]

        if len(my_lead_all) != 0:
            for my_lead in my_lead_all:
                channel = lead_in_data.index(my_lead)
                tmp_data = tmp_data_res[0][:, channel]
                idx_list = tmp_ann_res['sample']
                label_list = np.array(tmp_ann_res['symbol'])
                aux_list = np.array([i.strip('\x00') for i in tmp_ann_res['aux_note']])
                full_aux_list = [''] * tmp_data_res[1]['sig_len']  # 将aux扩展为完整长度
                for i in range(len(aux_list)):
                    full_aux_list[idx_list[i]] = aux_list[i]
                    if label_list[i] in ['[', '!']:
                        full_aux_list[idx_list[i]] = '(VF'  # 从心跳标签复制VF起始
                    if label_list[i] in [']']:
                        full_aux_list[idx_list[i]] = '(N'  # 从心跳标签复制VF结束

                # 填充空值
                for i in range(1, len(full_aux_list)):
                    if full_aux_list[i] == '':
                        full_aux_list[i] = full_aux_list[i - 1]
                full_aux_list = np.array(full_aux_list)

                idx_start = 0
                while idx_start <= len(tmp_data) - window_size:
                    idx_end = idx_start + window_size
                    tmp_label_beat = label_list[np.logical_and(idx_list >= idx_start, idx_list <= idx_end)]
                    tmp_label_rhythm = full_aux_list[idx_start:idx_end]
                    tmp_label = list(np.unique(tmp_label_beat)) + list(np.unique(tmp_label_rhythm))
                    tmp_label = get_label_map(tmp_label)

                    # 如果包含 'VF/VT' 标签，记录该文件名
                    if 'VF/VT' in tmp_label:
                        vfvt_records.append(record_name)
                        break  # 不再处理该记录，直接跳到下一条记录

                    idx_start += 2 * fs  # 步长为2秒

        else:
            print(f'No valid lead in {record_name}')
            continue

    return vfvt_records


# 调用该函数并打印含有'VF/VT'标签的记录名
path1 = '/Users/shiqissszj/Documents/Datasets/mitdb'  # 数据集路径
path2 = '/Users/shiqissszj/Documents/Datasets/cudb'  # 数据集路径
path3 = '/Users/shiqissszj/Documents/Datasets/vfdb'  # 数据集路径
vfvt_records = find_vfvt_records(path3)
print("Records with VF/VT labels:")
for record in vfvt_records:
    print(record)
