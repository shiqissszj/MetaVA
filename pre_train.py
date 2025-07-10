import numpy as np
from sklearn.model_selection import train_test_split
import numpy.random
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import os
from net1d import Net1D, MyDataset
from tran_models import ecgTransForm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader
import copy
import mdataprocess as dp
import util
import MAML
import args
import MTL
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.metrics import roc_auc_score  # 用于计算AUC

import sys

# data_root_path = '/data/myj/MetaLearner'
data_root_path = '/data/myj/MetaVA_V1'
save_model_dir = data_root_path + '/trained_modelsNet1D'

# 记录训练全内容
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def MAMLtrain(model, xset, yset, iterations, writer=None):
    xset, yset = dp.pair_shuffle(data=xset, label=yset)
    best_acc = best_auc = 0
    global_step = 0

    for innerlr in [5e-3]:
        for outerlr in [5e-3]:
            for update in [5]:
                print(f"[MAMLtrain] innerlr={innerlr}, outerlr={outerlr}, update={update}, iterations={iterations}")

                if writer is not None:
                    writer.add_text(
                        "HyperParams_MAML",
                        f"innerlr={innerlr}, outerlr={outerlr}, update={update}, iterations={iterations}",
                        global_step=global_step
                    )

                # 初始化
                raw_model = torch.load(save_model_dir + '/raw.pkl')
                meta_learner = MAML.MetaLearner(raw_model)

                model_cl = MAML.MAML(
                    meta_learner,
                    xset,
                    yset,
                    innerlr=innerlr,
                    outerlr=outerlr,
                    update=update,
                )

                device = torch.device('cuda:2' if torch.cuda.is_available() else "cpu")
                model_cl.to(device)

                optimizer = optim.Adam(model_cl.parameters(), lr=outerlr, weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

                for task_index in tqdm(range(iterations), desc='Task', leave=False):
                    # 将当前迭代task_index传给metatrain
                    sum_loss_tuple = model_cl.metatrain(iteration=task_index)

                    if isinstance(sum_loss_tuple, tuple):
                        sum_loss, train_acc, train_auc = sum_loss_tuple
                    else:
                        sum_loss = sum_loss_tuple
                        train_acc, train_auc = 0.0, 0.0

                    if torch.isnan(sum_loss).any():
                        print(f"[OuterLoop] sum_loss is NaN at iteration {task_index}!")
                        continue

                    optimizer.zero_grad()
                    sum_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model_cl.parameters(), max_norm=8.0)
                    optimizer.step()

                    scheduler.step(sum_loss)

                    print(
                        f"[OuterLoop] iteration={task_index}, sum_loss={sum_loss.item():.4f}, acc={train_acc:.4f}, auc={train_auc:.4f}\n")

                    if writer is not None:
                        writer.add_scalar("MAML/Train_Loss", sum_loss.item(), global_step=global_step)
                        writer.add_scalar("MAML/Train_Acc", train_acc, global_step=global_step)
                        writer.add_scalar("MAML/Train_AUC", train_auc, global_step=global_step)

                    global_step += 1

                    # # 每20个任务保存一次模型
                    # if (task_index + 1) % 2 == 0:
                    #     save_path = os.path.join(data_root_path, 'trained_models', f'metalearning_task_{task_index+1}.pkl')
                    #     torch.save(model_cl, save_path)
                    #     print(f"模型已保存于任务 {task_index+1}，路径：{save_path}")

                # 训练结束后验证
                auc, acc = model_cl.metavalid()
                print(f"[MAML] Validation => AUC: {auc:.4f}, Acc: {acc:.4f}")
                if writer is not None:
                    writer.add_scalar("MAML/Val_AUC", auc, global_step=global_step)
                    writer.add_scalar("MAML/Val_Acc", acc, global_step=global_step)

                if (auc + acc > best_acc + best_auc):
                    best_acc = acc
                    best_auc = auc
                    torch.save(model_cl, save_model_dir + '/metalearning.pkl')

    return


if __name__ == '__main__':
    # 创建 Tee 实例，将 stdout 和 stderr 同时写入控制台和文件
    log_file = open("training_log.txt", "w")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    # 1) 初始化 TensorBoard SummaryWriter
    exp_name = "exp{}".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=f'./tensorboard_logs/va_logs/{exp_name}')

    # 2) 读取数据
    train_data = np.load(data_root_path + '/data/train_data.npy', allow_pickle=True)
    train_label = np.load(data_root_path + '/data/train_label.npy', allow_pickle=True)
    # train_data = np.load(data_root_path + '/processed_data_vtac/train_data.npy', allow_pickle=True)
    # train_label = np.load(data_root_path + '/processed_data_vtac/train_label.npy', allow_pickle=True)
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    # 3) 初始化原始模型 并保存
    # model = Net1D(
    #     in_channels=1,
    #     base_filters=128,
    #     ratio=1.0,
    #     filter_list=[128, 64, 64, 32, 32, 16, 16],
    #     m_blocks_list=[2, 2, 2, 2, 2, 2, 2],
    #     kernel_size=16,
    #     stride=2,
    #     groups_width=16,
    #     verbose=False,
    #     n_classes=2,
    #     use_bn=False
    # ).cuda(2)
    model = ecgTransForm(
        input_channels=1,
        mid_channels=64,
        trans_dim=16,
        num_heads=4,
        dropout=0.5,
        num_classes=2,
        stride=2
    ).cuda(1)

    torch.save(model, save_model_dir + '/raw.pkl')

    # 4) 创建一个用于训练的模型
    # model_test = Net1D(
    #     in_channels=1,
    #     base_filters=128,
    #     ratio=1.0,
    #     filter_list=[128, 64, 64, 32, 32, 16, 16],
    #     m_blocks_list=[2, 2, 2, 2, 2, 2, 2],
    #     kernel_size=16,
    #     stride=2,
    #     groups_width=16,
    #     verbose=False,
    #     n_classes=2,
    #     use_bn=False
    # ).cuda(2)
    model_test = ecgTransForm(
        input_channels=1,
        mid_channels=64,
        trans_dim=16,
        num_heads=4,
        dropout=0.5,
        num_classes=2,
        stride=2
    ).cuda(1)

    model_test.load_state_dict(model.state_dict())

    # 5) MAML训练, 并记录到 TensorBoard
    MAMLtrain(model=model_test, xset=train_data, yset=train_label, iterations=300, writer=writer)

    print('save success')

    # 6) 训练完成后关闭 writer 和日志文件
    writer.close()
    log_file.close()