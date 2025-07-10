import numpy as np
import os
import wfdb
from collections import Counter
import pickle
import random
import sys
from tqdm import tqdm
from scipy.interpolate import interp1d

from scipy.signal import butter, lfilter
from tqdm import tqdm
from scipy.interpolate import interp1d
import mdataprocess as dp

flatten = lambda t: [item for sublist in t for item in sublist]

def bandpass_filter(data, lowcut, highcut, filter_order=4):
    nyquist_freq = 0.5 * data[1]['fs']

    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = butter(filter_order, [low, high], btype="band")
    for i in range(np.shape(data[0])[1]):
        data[0][:,i] = lfilter(b, a, data[0][:,i])
    return data

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


def resample_unequal(ts, fs_in, fs_out, t):
    """
    interploration
    """
    fs_in, fs_out = int(fs_in), int(fs_out)
    if fs_out == fs_in:
        return ts
    else:
        x_old = np.linspace(0, 1, num=fs_in * t, endpoint=True)
        x_new = np.linspace(0, 1, num=fs_out * t, endpoint=True)
        y_old = ts
        f = interp1d(x_old, y_old, kind='linear')
        y_new = f(x_new)
        return y_new


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

def preprocess_data(path, save_path, prefix):

    valid_lead = ['MLII', 'ECG'] # extract all similar leads
    fs_out = 200
    t = 2
    window_size_t = 2 # second
    stride_t = 2 # second

    test_ind = []
    all_pid = []
    all_data = []
    all_label = []

    with open(os.path.join(path, 'RECORDS'), 'r') as fin:
        all_record_name = fin.read().strip().split('\n')

    for record_name in all_record_name:
        cnt = 0
        try:
            tmp_ann_res = wfdb.rdann(path + '/' + record_name, 'atr').__dict__
            tmp_data_res = wfdb.rdsamp(path + '/' + record_name)
        except:
            print('read data failed')
            continue
        fs = tmp_data_res[1]['fs']
        window_size = int(fs*window_size_t)
        stride = int(fs*stride_t)
       # tmp_data_res = bandpass_filter(tmp_data_res, 0.5, 50)

        lead_in_data = tmp_data_res[1]['sig_name']

        my_lead_all = []
        for tmp_lead in valid_lead:
            if tmp_lead in lead_in_data:
                my_lead_all.append(tmp_lead)
        if len(my_lead_all) != 0:
            for my_lead in my_lead_all:
                pp_pid = []
                pp_data = []
                pp_label = []
                channel = lead_in_data.index(my_lead)
                tmp_data = tmp_data_res[0][:, channel]
                idx_list = tmp_ann_res['sample']
                label_list = np.array(tmp_ann_res['symbol'])
                aux_list = np.array([i.strip('\x00') for i in tmp_ann_res['aux_note']])
                full_aux_list = [''] * tmp_data_res[1]['sig_len'] # expand aux to full length
                for i in range(len(aux_list)):
                    full_aux_list[idx_list[i]] = aux_list[i] # copy old aux
                    if label_list[i] in ['[', '!']:
                        full_aux_list[idx_list[i]] = '(VF' # copy VF start from beat labels
                    if label_list[i] in [']']:
                        full_aux_list[idx_list[i]] = '(N' # copy VF end from beat labels
                for i in range(1,len(full_aux_list)):
                    if full_aux_list[i] == '':
                        full_aux_list[i] = full_aux_list[i-1] # copy full_aux_list from itself, fill empty strings
                full_aux_list = np.array(full_aux_list)
                idx_start = 0

                while idx_start <= len(tmp_data) - window_size:
                    idx_end = idx_start+window_size
                    tmpdata = resample_unequal(tmp_data[idx_start:idx_end], fs, fs_out, t)
                    if not -100 < np.mean(tmpdata) < 100:
                        idx_start += fs
                        continue
                    pp_pid.append("{}".format(record_name))

                    pp_data.append(resample_unequal(tmp_data[idx_start:idx_end], fs, fs_out, t))
                    tmp_label_beat = label_list[np.logical_and(idx_list>=idx_start, idx_list<=idx_end)]
                    tmp_label_rhythm = full_aux_list[idx_start:idx_end] # be careful
                    tmp_label = list(np.unique(tmp_label_beat))+list(np.unique(tmp_label_rhythm))
                    tmp_label = get_label_map(tmp_label)
                    if 'VF/VT' in tmp_label and cnt <= 150:
                        idx_start += int(0.1 * fs)
                        cnt += 1
                    else:
                        idx_start += 2 * fs
                    pp_label.append(tmp_label)


                all_pid.extend(pp_pid)
                all_data.extend(pp_data)
                all_label.extend(pp_label)

                print('record_name:{}, len:{}, lead:{}, fs:{}, count:{}, labels:{}'.format(record_name, tmp_data_res[1]['sig_len'], my_lead, fs, len(pp_data), Counter(flatten(pp_label))))

        else:
            print('lead in data: [{0}]. no valid lead in {1}'.format(lead_in_data, record_name))
            continue

    all_pid = np.array(all_pid)
    all_data = np.array(all_data)
    # if prefix == 'mitdb':
    #     all_label = [x[0] if isinstance(x, list) and len(x) > 0 else x for x in all_label]
    # all_label = [x[0] for x in all_label]  # 只取每个列表的第一个元素
    all_label = [x[0] if isinstance(x, list) and len(x) > 0 else "Others" for x in all_label]

    # 定义映射关系
    label_mapping = {'VF/VT': 1, 'Others': 0}

    # 先确保 all_label 结构一致，转换为字符串
    print("First 10 labels before mapping:", all_label[:10])
    print("Label types:", [type(x) for x in all_label[:10]])
    print("Unique labels before mapping:", np.unique(all_label))

    # 映射为 0 和 1
    all_label = [label_mapping.get(x, 0) for x in all_label]  # 如果有未知标签，设为 -1

    # 转换为 NumPy 数组
    all_label = np.array(all_label)
    print(all_label)
    print(all_pid.shape, all_data.shape)
    # print(Counter(flatten(all_label)))
    # print(Counter([tuple(_) for _ in all_label]))
    np.save(os.path.join(save_path, '{}_pid.npy'.format(prefix)), all_pid)
    np.save(os.path.join(save_path, '{}_data.npy'.format(prefix)), all_data)
    np.save(os.path.join(save_path, '{}_label.npy'.format(prefix)), all_label)
    print('{} done'.format(prefix))


if __name__ == "__main__":
    save_path = 'data0/'
    source = 0
    if not source:
        path = '/Users/shiqissszj/Documents/Datasets/mitdb'
        # path='/data/myj/mitdbcudb/mitdb'
        prefix = 'mitdb'
        preprocess_data(path, save_path, prefix)

        path = '/Users/shiqissszj/Documents/Datasets/vfdb'
        prefix = 'vfdb'
        preprocess_data(path, save_path, prefix)

        # path = '/data/myj/mitdbcudb/cudb'
        path = '/Users/shiqissszj/Documents/Datasets/cudb'
        prefix = 'cudb'
        preprocess_data(path, save_path, prefix)

    train_data, adapt_data, train_label, adapt_label = dp.create_sets(filenames=['mitdb', 'cudb'])
    np.save('data/train_data.npy', train_data)
    np.save('data/adapt_data.npy', adapt_data)
    np.save('data/train_label.npy', train_label)
    np.save('data/adapt_label.npy', adapt_label)