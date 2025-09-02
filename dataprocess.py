import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.metrics import auc
from collections import Counter


def read_data(filename):
    '''
    Description:
    Read dataset and return data and labels grouped by patient (pid).
    :param filename: dataset name
    :return: data and labels grouped by patient (pid)
    '''

    Data = dict()  # store each patient's ECG data
    Label = dict()  # store each patient's labels
    data = np.load(f'data/{filename}_data.npy')  # load ECG data
    pid = np.load(f'data/{filename}_pid.npy')  # load patient IDs
    label = np.load(f'data/{filename}_label.npy', allow_pickle=True)  # load labels

    name = np.unique(pid)
    for i in name:
        # For each patient, group by pid and store ECG data and labels
        Data[i] = data[pid == i]
        Label[i] = label[pid == i]

    return Data, Label


def create_sets(filenames, test_split=0.2):
    '''
    Create train and test sets.
    :param filenames: dataset name(s) (string or list of strings)
    :param test_split: test set ratio, default 20%.
    :return: train/test data and labels
    '''

    Data = []
    Label = []
    patient_ids = []

    if isinstance(filenames, str):
        filenames = [filenames]

    # Iterate through all files, read and merge
    for filename in filenames:
        data, label = read_data(filename)
        for pid, ecg_data in data.items():
            patient_ids.append(pid)
            Data.append(ecg_data)
            Label.append(label[pid])

    Data = np.array(Data, dtype=object)
    Label = np.array(Label, dtype=object)

    Data_filtered = []
    Label_filtered = []
    cnt = cnt_ep = 0

    # Normalize data
    for i in range(len(Data)):
        tmp_data = Data[i]
        tmp_label = Label[i]

        # Check if data contains NaN or Inf
        if np.any(np.isnan(tmp_data)) or np.any(np.isinf(tmp_data)):
            print("Warning: Found NaN or Inf in input data before normalization!")
            print(f"Data is: {tmp_data}, the shape of it is {tmp_data.shape}")
            cnt += 1
            continue

        tmp_std = np.std(tmp_data)
        tmp_mean = np.mean(tmp_data)

        # Handle near-zero std case
        epsilon = 1e-8
        if tmp_std < epsilon:
            print("Warning: Found near-zero std, adding epsilon to avoid division by zero.")
            tmp_std = epsilon
            cnt_ep += 1

        normalized_data = (tmp_data - tmp_mean) / tmp_std

        Data_filtered.append(normalized_data)
        Label_filtered.append(tmp_label)

    print(f"The sum of NaN/Inf data skipped: {cnt}")
    print(f"The sum of near-zero std data adjusted: {cnt_ep}")

    Data_filtered = np.array(Data_filtered, dtype=object)
    Label_filtered = np.array(Label_filtered, dtype=object)

    num_samples = len(Data_filtered)
    split_index = int(num_samples * (1 - test_split))

    XTrain = Data_filtered[:split_index]
    YTrain = Label_filtered[:split_index]
    XTest = Data_filtered[split_index:]
    YTest = Label_filtered[split_index:]

    return XTrain, XTest, YTrain, YTest


def pair_shuffle(data, label):
    index = np.random.permutation(len(label))
    return data[index], label[index]


def FilterNwaysKshots(data, label, N, train_shots, test_shots=1, remain=False):
    '''
    Description:
    Randomly pick N-way K-shot data from the whole set
    :param data: data
    :param label: label
    :param N: number of classes
    :param train_shots: number of shots each class in train_set
    :param test_shots: number of shots each class in test_set
    :param remain: whether to return the remaining data
    :return: train_set, test_set, (remaining data)
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

    # Check the shape of label
    if isinstance(label, np.ndarray) and label.ndim == 0:
        label = np.expand_dims(label, 0)  # convert to 1D array
    elif not isinstance(label, np.ndarray):
        label = np.array(label)

    # print("Entering FilterNwaysKshots, label shape:", label.shape)

    for i in name[0: N]:
        is_name = label == i
        l, d = label[is_name], data[is_name]
        print("Label distribution:", Counter(l))
        # if not len(l) >= train_shots + test_shots:
        #     print(f"Class {i} has insufficient samples: {len(l)}, need at least {train_shots + test_shots}")
        #     raise IndexError("dataprocess: FilterNwaysKshots: insufficient data for some class")
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


from sklearn.metrics import roc_auc_score


def roc_curve(prob, label):
    '''
    Description:
    Calculate the AUC using scikit-learn's roc_auc_score
    :param prob: probability of positive class
    :param label: corresponding labels
    :return: AUC
    '''
    try:
        auc_value = roc_auc_score(label, prob)
    except ValueError as e:
        print()
        print(f"AUC Calculation Error: {e}")
        auc_value = 0.0
    return auc_value
