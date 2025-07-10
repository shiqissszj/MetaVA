import numpy as np
from sklearn.model_selection import train_test_split
import numpy.random
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import os
import csv
from net1d import Net1D, MyDataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, f1_score
import MAML
import mdataprocess as dp

# model_src = '/data/myj/MetaVA_MAML'
model_src = '/data/myj/MetaVA_V1'

def test_para(para, data, label, update, txt_log_path, csv_log_path):
    AUC_array = []
    ACC_array = []
    all_losses = []  # 添加存储所有患者loss的列表

    # Iterate over each data entry
    for i in tqdm(range(len(data)), desc=para + '_Task'):
        auc = acc = pr_auc = f1 = 0

        labels_i = label[i]
        pos_idx = np.where(labels_i == 1)[0]
        neg_idx = np.where(labels_i == 0)[0]

        choose_pos = np.random.choice(pos_idx, min(30, len(pos_idx)), replace=False)
        choose_neg = np.random.choice(neg_idx, min(30, len(neg_idx)), replace=False)
        train_index = np.hstack([choose_pos, choose_neg])
        np.random.shuffle(train_index)

        # Fine-tuning
        for lr in [5e-3]:
            model = torch.load(os.path.join(model_src, 'trained_models', para + '.pkl'))
            if para == 'metalearning':
                model = model.model

            # Fine-tuning
            temauc, temacc, pred_prob, acc_label, patient_losses = fine_tune(model=model, data=np.array(data[i]),
                                                             label=np.array(label[i]), lr=lr,
                                                             classes=2, n_epoch=update, train_size=16,
                                                             train_index=train_index)
            all_losses.append(patient_losses)  # 记录当前患者的loss
            auc = max(auc, temauc)
            acc = max(acc, temacc)
            # Calculate PR-AUC and F1-Score
            pr_auc = average_precision_score(acc_label, pred_prob)
            f1 = f1_score(acc_label, [1 if x > 0.5 else 0 for x in pred_prob])
            print(
                f"The results for Finetune are AUC:{temauc:.4f}, ACC:{temacc:.4f}, PR-AUC:{pr_auc:.4f}, F1:{f1:.4f}.")

        # Store results in arrays
        AUC_array.append(auc)
        ACC_array.append(acc)

        # Append results to TXT log after each data entry
        with open(txt_log_path, 'a') as f:
            f.write(f"Model={para}, Data Index={i}, AUC_fine_tune={auc}, ACC_fine_tune={acc}, PR_AUC={pr_auc}, F1={f1}\n")

        # Append results to CSV log after each data entry
        with open(csv_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([para, i, auc, acc, pr_auc, f1])

    # 保存所有患者的loss到单独的CSV文件
    loss_csv_path = csv_log_path.replace('.csv', '_losses.csv')
    with open(loss_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Patient', 'Epoch', 'Loss'])  # 写入表头
        for patient_idx, patient_losses in enumerate(all_losses):
            for epoch, loss in enumerate(patient_losses):
                writer.writerow([patient_idx, epoch + 1, loss])

    # Calculate average AUC, ACC
    if para == 'metalearning':
        avg_auc = sum(AUC_array) / len(AUC_array)
        avg_acc = sum(ACC_array) / len(ACC_array)

        # Log average results to the TXT file
        with open(txt_log_path, 'a') as f:
            f.write(f"\n{para} average AUC: {avg_auc}, Average ACC: {avg_acc}\n")

        return avg_auc, avg_acc
    else:
        return sum(AUC_array) / len(AUC_array), sum(ACC_array) / len(ACC_array)

def fine_tune(model, data, label, lr, classes, n_epoch, train_size=5, batch_size=1, train_index=list(range(10))):
    '''
    ** Description **
    The main part of fine tune(quickly adaptation)
    :param model:
    :param data:
    :param label:
    :param lr:
    :param classes:
    :param n_epoch:
    :param train_size:
    :param batch_size:
    :return:
    '''
    data = np.expand_dims(data, 1)
    train_data = data[train_index]
    train_label = label[train_index]
    test_data = data
    test_label = label

    initial_lr = lr * 0.1
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    loss_func = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda:3')
    model.to(device)
    
    # 早停机制
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    min_delta = 0.001
    
    # 记录loss
    epoch_losses = []
    
    # train
    model.train()
    step = 1
    while step <= n_epoch:
        indices = np.random.permutation(len(train_data))
        train_data_shuffled = train_data[indices]
        train_label_shuffled = train_label[indices]
        
        dataset_train = MyDataset(train_data_shuffled, train_label_shuffled)
        dataset_test = MyDataset(test_data, test_label)
        trainset = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        testset = DataLoader(dataset_test, batch_size=batch_size)
        
        sum_loss = 0
        model.train()
        for batch_idx, batch in enumerate(trainset):
            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            loss = loss_func(pred, input_y)
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            sum_loss += loss
            
        current_loss = sum_loss.item()
        epoch_losses.append(current_loss)

        print(f"Step {step}, Loss: {current_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 早停检查
        if current_loss < best_loss - min_delta:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered at step {step}")
            break
            
        # 如果loss异常，提前停止
        if current_loss > 1000:
            print(f"Loss too high ({current_loss:.4f}), stopping training")
            break
            
        scheduler.step(current_loss)
        step += 1

    # test
    model.eval()
    prog_iter_test = tqdm(testset, desc="Testing", leave=False)
    pred_prob = []
    acc_label = []
    acc = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(prog_iter_test):
            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            pred = F.softmax(pred, dim=1)
            pred_prob.append(pred[:, 1])
            acc += sum(pred.argmax() == input_y)
            acc_label.extend(list(int(i) for i in input_y))
    AUC = dp.roc_curve(
        np.array([p.cpu().numpy() for p in pred_prob]),
        np.array(acc_label)
    )
    acc = acc / len(test_label)

    return AUC, acc, np.array([p.cpu().numpy() for p in pred_prob]), acc_label, epoch_losses

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    adapt_data = np.load('/data/myj/MetaVA_V1/dataHV/adapt_data.npy', allow_pickle=True)
    adapt_label = np.load('/data/myj/MetaVA_V1/dataHV/adapt_label.npy', allow_pickle=True)
    # adapt_data = np.load('/data/myj/MetaVA_V1/data0/adapt_data.npy', allow_pickle=True)
    # adapt_label = np.load('/data/myj/MetaVA_V1/data0/adapt_label.npy', allow_pickle=True)

    # idx_list = [10, 17, 29, 32]
    # adapt_data = adapt_data[idx_list]
    # adapt_label = adapt_label[idx_list]

    netmodel = ['metalearning']  # 可以根据需求扩展模型类型，如 ['metalearning', 'traditional']

    txt_log_path = './dataHV/single.txt'
    csv_log_path = './dataHV/single.csv'

    # 初始化txt日志文件
    with open(txt_log_path, 'w') as f:
        f.write("开始记录训练/测试日志...\n")

    # 初始化CSV日志文件
    with open(csv_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Model", "Data Index", "AUC_fine_tune", "ACC_fine_tune", "PR_AUC", "F1"])

    result = {}

    for model_type in netmodel:
        mean_auc = []
        mean_acc = []

        auc, acc = test_para(para=model_type, data=adapt_data, label=adapt_label, update=300,
                                         txt_log_path=txt_log_path, csv_log_path=csv_log_path)

        mean_auc.append(auc)
        mean_acc.append(acc)

        # 计算当前模型类型的平均结果
        avg_auc = sum(mean_auc) / len(mean_auc)
        avg_acc = sum(mean_acc) / len(mean_acc)

        # 记录每个模型类型的平均结果
        with open(txt_log_path, 'a') as f:
            f.write(f"\n{model_type} average AUC: {avg_auc}, Average ACC: {avg_acc}\n")

        # 存储最终结果
        result[model_type + 'auc'] = mean_auc
        result[model_type + 'acc'] = mean_acc

    print(result)
    exit(0) 