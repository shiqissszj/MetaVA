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
    AUC_array2 = []
    ACC_array2 = []
    all_losses = []

    # Iterate over each data entry
    for i in tqdm(range(len(data)), desc=para + '_Task'):
        auc1 = acc1 = pr_auc1 = f1_1 = 0

        labels_i = label[i]
        pos_idx = np.where(labels_i == 1)[0]
        neg_idx = np.where(labels_i == 0)[0]

        choose_pos = np.random.choice(pos_idx, min(30, len(pos_idx)), replace=False)
        choose_neg = np.random.choice(neg_idx, min(30, len(neg_idx)), replace=False)
        train_index = np.hstack([choose_pos, choose_neg])
        np.random.shuffle(train_index)

        for lr in [5e-3]:
            model = torch.load(os.path.join(model_src, 'trained_models', para + '.pkl'))
            if para == 'metalearning':
                model = model.model

            # Re-load model for second method (fine_tune2：pre-fine-tune)
            model = torch.load(os.path.join(model_src, 'trained_models', para + '.pkl'))
            if para == 'metalearning':
                model = model.model

            if para == 'metalearning':
                temauc, temacc, pred_prob, acc_label, patient_losses = fine_tune2(model=model, data=np.array(data[i]),
                                                                  label=np.array(label[i]),
                                                                  lr=lr, classes=2, n_epoch=update, train_size=16,
                                                                  train_index=train_index)
                all_losses.append(patient_losses)
                auc1 = max(auc1, temauc)
                acc1 = max(acc1, temacc)
                # Calculate PR-AUC and F1-Score
                pr_auc1 = average_precision_score(acc_label, pred_prob)
                f1_1 = f1_score(acc_label, [1 if x > 0.5 else 0 for x in pred_prob])
                print(
                    f"The results for Finetune2 are AUC:{temauc:.4f}, ACC:{temacc:.4f}, PR-AUC:{pr_auc1:.4f}, F1:{f1_1:.4f}.")

        AUC_array2.append(auc1)
        ACC_array2.append(acc1)

        # Append results to TXT log after each data entry
        with open(txt_log_path, 'a') as f:
            f.write(f"Model={para}, Data Index={i}, "
                    f"AUC_fine_tune2={auc1}, ACC_fine_tune2={acc1}, PR_AUC1={pr_auc1}, F1={f1_1}\n")

        # Append results to CSV log after each data entry
        with open(csv_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([para, i, auc1, acc1, pr_auc1, f1_1])

    # 保存所有患者的loss到单独的CSV文件
    loss_csv_path = csv_log_path.replace('.csv', '_losses.csv')
    with open(loss_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Patient', 'Epoch', 'Loss'])
        for patient_idx, patient_losses in enumerate(all_losses):
            for epoch, loss in enumerate(patient_losses):
                writer.writerow([patient_idx, epoch + 1, loss])

    # Calculate average AUC, ACC
    if para == 'metalearning':
        avg_auc1 = sum(AUC_array2) / len(AUC_array2)
        avg_acc1 = sum(ACC_array2) / len(ACC_array2)

        # Log average results to the TXT file
        with open(txt_log_path, 'a') as f:
            f.write(f"\n{para} Average AUC1: {avg_auc1}, Average ACC1: {avg_acc1}\n")

        return avg_auc1, avg_acc1
    else:
        return 0, 0


def fine_tune2(model, data, label, lr, classes, n_epoch, train_size=5, batch_size=1, train_index=list(range(10))):
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
    
    # 初始学习率
    initial_lr = lr * 0.1
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    loss_func = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda:3')
    model.to(device)
    model.train()
    
    # 添加早停机制
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    min_delta = 0.001
    
    # 记录训练过程中的loss
    epoch_losses = []
    
    for tt in tqdm(range(30), desc="epoch", leave=False):
        # 随机打乱训练数据
        indices = np.random.permutation(len(train_data))
        train_data_shuffled = train_data[indices]
        train_label_shuffled = train_label[indices]
        
        dataset_train = MyDataset(train_data_shuffled[: 20], train_label_shuffled[: 20])
        dataset_valid = MyDataset(train_data_shuffled[20:], train_label_shuffled[20:])
        trainset = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        validset = DataLoader(dataset_valid, batch_size=batch_size)
        
        sum_loss = 0
        learner = model.clone()
        for _ in tqdm(range(5), desc='update_num', leave=False, colour='white'):
            learner.train()

            prog_iter = tqdm(trainset, desc='Epoch', leave=False, colour='yellow')
            for batch_idx, batch in enumerate(prog_iter):
                input_x, input_y = tuple(t.to(device) for t in batch)
                pred = learner(input_x)
                loss = loss_func(pred, input_y)

                torch.nn.utils.clip_grad_norm_(learner.parameters(), max_norm=1.0)
                
                learner.adapt(loss=loss / batch_size)
            
            # test
            learner.eval()
            test_task = tqdm(validset, desc="Test")
            for batch_idx, batch in enumerate(test_task):
                x, y = tuple(t.to(device) for t in batch)
                pred = model(x)
                loss = loss_func(pred, y)
                sum_loss += loss
                
        sum_loss /= train_size
        current_loss = sum_loss.item()
        epoch_losses.append(current_loss)
        
        # 早停检查
        if current_loss < best_loss - min_delta:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {tt+1}")
            break
            
        optimizer.zero_grad()
        sum_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step(current_loss)

        print(f"Epoch {tt+1}, Loss: {current_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    dataset_train = MyDataset(train_data, train_label)
    dataset_test = MyDataset(test_data, test_label)
    trainset = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    testset = DataLoader(dataset_test, batch_size=batch_size)
    
    # 使用较小的学习率进行微调
    optimizer = optim.Adam(model.parameters(), lr=initial_lr * 0.1, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    step = 1
    best_loss = float('inf')
    patience_counter = 0

    # train
    model.train()
    while step <= n_epoch:
        step += 1
        sum_loss = 0
        for batch_idx, batch in enumerate(trainset):
            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            loss = loss_func(pred, input_y)
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            sum_loss += loss
            
        current_loss = sum_loss.item()
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
            
        scheduler.step(current_loss)
        
        # 如果loss异常，提前停止
        if current_loss > 1000:
            print(f"Loss too high ({current_loss:.4f}), stopping training")
            break

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
            acc += sum(pred.argmax() == input_y)
            pred_prob.append(pred[:, 1])
            acc_label.extend(list(int(i) for i in input_y))
    acc = acc / len(test_label)
    AUC = dp.roc_curve(
        np.array([p.cpu().numpy() for p in pred_prob]),
        np.array(acc_label)
    )

    return AUC, acc, np.array([p.cpu().numpy() for p in pred_prob]), acc_label, epoch_losses


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    # adapt_data = np.load('/data/myj/MetaVA_V1/dataHV/adapt_data.npy', allow_pickle=True)
    # adapt_label = np.load('/data/myj/MetaVA_V1/dataHV/adapt_label.npy', allow_pickle=True)
    adapt_data = np.load('/data/myj/MetaVA_V1/data0/adapt_data.npy', allow_pickle=True)
    adapt_label = np.load('/data/myj/MetaVA_V1/data0/adapt_label.npy', allow_pickle=True)

    idx_list = [10, 17, 29, 32]
    adapt_data = adapt_data[idx_list]
    adapt_label = adapt_label[idx_list]

    netmodel = ['metalearning']  # 可以根据需求扩展模型类型，如 ['metalearning', 'traditional']

    # 日志路径
    txt_log_path = './data0/1729.txt'
    csv_log_path = './data0/1729.csv'

    # 初始化txt日志文件
    with open(txt_log_path, 'w') as f:
        f.write("开始记录训练/测试日志...\n")

    # 初始化CSV日志文件
    with open(csv_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Model", "Data Index", "AUC_fine_tune2", "ACC_fine_tune2", "PR_AUC2", "F1_2"])

    result = {}

    for model_type in netmodel:
        mean_auc1 = []
        mean_acc1 = []

        # 直接调用 test_para 函数进行训练和测试
        auc1, acc1 = test_para(para=model_type, data=adapt_data, label=adapt_label, update=300,
                                         txt_log_path=txt_log_path, csv_log_path=csv_log_path)

        # 结果
        if auc1 != 0:
            mean_auc1.append(auc1)
            mean_acc1.append(acc1)

        # 计算当前模型类型的平均结果
        avg_auc1 = sum(mean_auc1) / len(mean_auc1) if mean_auc1 else 0
        avg_acc1 = sum(mean_acc1) / len(mean_acc1) if mean_acc1 else 0

        # 记录每个模型类型的平均结果
        with open(txt_log_path, 'a') as f:
            f.write(f"\n{model_type} Average AUC1: {avg_auc1}, Average ACC1: {avg_acc1}\n")

        # 存储最终结果
        result[model_type + 'newauc'] = mean_auc1
        result[model_type + 'newacc'] = mean_acc1

    print(result)
    exit(0)
