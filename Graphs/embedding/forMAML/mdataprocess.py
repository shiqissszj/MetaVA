import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import auc
from collections import Counter


def check(label, database):
    '''
    检查数据是否满足要求
    '''
    limit = 50  # 设置单一阈值
    if sum('VF/VT' in i for i in label) >= limit:
        return True
    return False


def read_data(filename):
    '''
    ** Description **
    读取vtac数据集并返回按患者（pid）分组的数据和标签。
    :param filename: 数据集名称（这里是'vtac'）
    :return: 按患者（pid）分组的数据和标签
    '''

    Data = dict()  # 存储每个患者的ECG数据
    Label = dict()  # 存储每个患者的标签
    data = np.load(f'data/{filename}_data.npy')  # 读取ECG数据
    pid = np.load(f'data/{filename}_pid.npy')  # 读取患者ID
    label = np.load(f'data/{filename}_label.npy', allow_pickle=True)  # 读取标签（True/False）

    name = np.unique(pid)  # 获取所有唯一的患者ID
    for i in name:
        # 对每个患者，按pid分组，存储患者的ECG数据和标签
        Data[i] = data[pid == i]
        Label[i] = label[pid == i]

    return Data, Label


def create_sets(filenames, test_split=0.2):
    '''
    创建训练集和测试集。
    :param filenames: 数据集名称（单个字符串或字符串数组）
    :param test_split: 测试集占比，默认 20%。
    :return: 训练集和测试集数据及标签
    '''

    Data = []  # 存储所有患者的ECG数据
    Label = []  # 存储所有患者的标签
    patient_ids = []  # 存储所有患者的ID（pid）

    # 如果传入的是单个字符串，转换为列表以统一处理
    if isinstance(filenames, str):
        filenames = [filenames]

    # 遍历所有文件，读取数据并合并
    for filename in filenames:
        data, label = read_data(filename)
        for pid, ecg_data in data.items():
            patient_ids.append(pid)
            Data.append(ecg_data)  # 存储 ECG 数据
            Label.append(label[pid])  # 存储标签

    # 转换为 NumPy 数组
    Data = np.array(Data, dtype=object)  # 每个元素是一个患者的ECG数据
    Label = np.array(Label, dtype=object)  # 每个元素是一个患者的标签

    # 用新的列表来存放过滤后的数据
    Data_filtered = []
    Label_filtered = []
    cnt = cnt_ep = 0

    # 对数据进行标准化处理
    for i in range(len(Data)):
        tmp_data = Data[i]
        tmp_label = Label[i]

        # 检查数据是否包含 NaN 或 Inf
        if np.any(np.isnan(tmp_data)) or np.any(np.isinf(tmp_data)):
            print("Warning: Found NaN or Inf in input data before normalization!")
            print(f"Data is: {tmp_data}, the shape of it is {tmp_data.shape}")
            cnt += 1
            continue  # 跳过该条数据

        tmp_std = np.std(tmp_data)
        tmp_mean = np.mean(tmp_data)

        # 处理标准差为 0 的情况（加小 epsilon）
        epsilon = 1e-8
        if tmp_std < epsilon:
            print("Warning: Found near-zero std, adding epsilon to avoid division by zero.")
            tmp_std = epsilon
            cnt_ep += 1

        # 进行标准化
        normalized_data = (tmp_data - tmp_mean) / tmp_std

        # 存入过滤后列表
        Data_filtered.append(normalized_data)
        Label_filtered.append(tmp_label)

    print(f"The sum of NaN/Inf data skipped: {cnt}")
    print(f"The sum of near-zero std data adjusted: {cnt_ep}")

    # 转换为 NumPy 数组
    Data_filtered = np.array(Data_filtered, dtype=object)
    Label_filtered = np.array(Label_filtered, dtype=object)

    # 划分训练集和测试集
    num_samples = len(Data_filtered)
    split_index = int(num_samples * (1 - test_split))

    XTrain = Data_filtered[:split_index]  # 训练集
    YTrain = Label_filtered[:split_index]
    XTest = Data_filtered[split_index:]  # 测试集
    YTest = Label_filtered[split_index:]

    return XTrain, XTest, YTrain, YTest


def pair_shuffle(data, label):
    index = np.random.permutation(len(label))
    return data[index], label[index]


def FilterNwaysKshots(data, label, N, train_shots, test_shots=1, remain=False):
    '''
    ** Description **
    randomly pick N-way K-shot data from the whole set
    :param data: data
    :param label: label
    :param N: number of classes
    :param train_shots: number of shots each class in train_set
    :param test_shots: number of shots each class in test_set
    :param remain: whether to return the remaining data
    :return:train_set, test_set, (remaining data)
    '''
    # data, label must be ndarray
    name = np.unique(label)
    np.random.shuffle(name)
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    remain_x = []
    remain_y = []

    # 检查 label 的形状
    if isinstance(label, np.ndarray) and label.ndim == 0:
        label = np.expand_dims(label, 0)  # 转换为一维数组
    elif not isinstance(label, np.ndarray):
        label = np.array(label)

    # print("进入 FilterNwaysKshots 函数，label 的形状: ", label.shape)

    for i in name[0: N]:
        is_name = label == i
        l, d = label[is_name], data[is_name]
        print("label[i]数据分布: ", Counter(l))
        # if not len(l) >= train_shots + test_shots:
        #     print(f"类别 {i} 的样本数不足: {len(l)}, 需要至少 {train_shots + test_shots}")
        #     raise IndexError("dataprocess: FilterNwaysKshots: we lack some class of data")
        index = np.random.permutation(len(l))
        train_y.extend(l[index[: train_shots]])
        train_x.extend(d[index[: train_shots]])
        test_x.extend(d[index[train_shots: train_shots + test_shots]])
        test_y.extend(l[index[train_shots: train_shots + test_shots]])
        remain_x.extend(d[index[train_shots + test_shots:]])
        remain_y.extend(l[index[train_shots + test_shots:]])
    # print("train_data shape:", train_x.shape, "dtype:", train_x.dtype)
    # print("train_label shape:", train_y.shape, "dtype:", train_y.dtype)

    if remain == False:
        return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

    for i in name[N:]:
        remain_x.extend(data[label == i])
        remain_y.extend(label[label == i])
    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y), np.array(remain_x), np.array(
        remain_y)


# def FilterNwaysKshots(data, label, N, train_shots, test_shots = 1, remain = False):
#     '''
#     ** Description **
#     randomly pick N-way K-shot data from the whole set
#     :param data: data
#     :param label: label
#     :param N: number of classes
#     :param train_shots: number of shots each class in train_set
#     :param test_shots: number of shots each class in test_set
#     :param remain: whether to return the remaining data
#     :return:train_set, test_set, (remaining data)
#     '''
#     # data, label must be ndarray
#     name = np.unique(label)
#     np.random.shuffle(name)
#     train_x = []
#     train_y = []
#     test_x = []
#     test_y = []
#     remain_x = []
#     remain_y = []
#     for i in name[0 : N]:
#         is_name = label == i
#         l, d = label[is_name], data[is_name]
#         if not len(l) >= train_shots + test_shots:
#             raise IndexError("dataprocess: FilterNwaysKshots: we lack some class of data")
#         index = np.random.permutation(len(l))
#         train_y.extend(l[index[: train_shots]])
#         train_x.extend(d[index[: train_shots]])
#         test_x.extend(d[index[train_shots : train_shots + test_shots]])
#         test_y.extend(l[index[train_shots : train_shots + test_shots]])
#         remain_x.extend(d[index[train_shots + test_shots : ]])
#         remain_y.extend(l[index[train_shots + test_shots : ]])
#
#     if remain == False:
#         return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)
#
#     for i in name[N : ]:
#         remain_x.extend(data[label == i])
#         remain_y.extend(label[label == i])
#     return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y), np.array(remain_x), np.array(remain_y)

# def calc_rate(prob, label, threshold):
#     all_number = len(prob)
#     TP = FP = FN = TN = 0
#     for i in range(all_number):
#         if prob[i] > threshold:
#             if label[i] == 1:
#                 TP += 1
#             else:
#                 FP += 1
#         else:
#             if label[i] == 0:
#                 TN += 1
#             else:
#                 FN += 1
#     accracy = (TP + TN) / all_number
#     if TP + FP == 0:
#         precision = 0
#     else:
#         precision = TP / (TP + FP)
#     TPR = 1 if TP + FN == 0 else TP / (TP + FN)
#     TNR = 1 if TN == 0 else TN / (FP + TN)
#     FNR = 0 if FN == 0 else FN / (TP + FN)
#     # FPR = FP / (FP + TN)
#     if (FP + TN) == 0:
#         FPR = None
#     else:
#         FPR = FP / (FP + TN)
#     return accracy, precision, TPR, TNR, FNR, FPR

def calc_rate(prob, label, threshold):
    all_number = len(prob)
    TP = FP = FN = TN = 0
    for i in range(all_number):
        if prob[i] > threshold:
            if label[i] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if label[i] == 0:
                TN += 1
            else:
                FN += 1
    accuracy = (TP + TN) / all_number
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 1
    TNR = TN / (FP + TN) if (FP + TN) > 0 else 0  # 修正逻辑错误
    FNR = FN / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else None
    return accuracy, precision, TPR, TNR, FNR, FPR


# def roc_curve(prob, label):
#     '''
#     ** Description **
#     Draw roc curve and calculate the AUC
#     **
#     :param prob: prob of positive class
#     :param label: corresponding label
#     :return: AUC
#     '''
#     threshold_vaule = sorted(prob)
#     threshold_num = len(threshold_vaule)
#     accracy_array = np.zeros(threshold_num)
#     precision_array = np.zeros(threshold_num)
#     TPR_array = np.zeros(threshold_num)
#     FPR_array = np.zeros(threshold_num)
#     for thres in range(threshold_num):
#         accracy, precision, TPR, _, _, FPR = calc_rate(prob, label, threshold_vaule[thres])
#         if FPR is None:
#             continue
#         accracy_array[thres] = accracy
#         precision_array[thres] = precision
#         TPR_array[thres] = TPR
#         FPR_array[thres] = FPR
#     AUC = auc(FPR_array, TPR_array)
#     # plt.plot(FPR_array, TPR_array)
#     # plt.title('roc')
#     # plt.xlabel('FPR_array')
#     # plt.ylabel('TPR_array')
#     # plt.show()
#     return AUC

from sklearn.metrics import roc_auc_score


def roc_curve(prob, label):
    '''
    ** Description **
    Calculate the AUC using scikit-learn's roc_auc_score
    **
    :param prob: prob of positive class
    :param label: corresponding label
    :return: AUC
    '''
    try:
        auc_value = roc_auc_score(label, prob)
    except ValueError as e:
        print()
        print(f"AUC Calculation Error: {e}")
        auc_value = 0.0
    return auc_value
