import numpy as np
from sklearn.model_selection import train_test_split
import numpy.random
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import os
from tran_models import ecgTransForm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader
import copy
import dataprocess as dp
import util
import MAML
import args
from datetime import datetime
from sklearn.metrics import roc_auc_score  # for computing AUC

import sys


# split by individual and class
def split_data_by_individual_and_class(data, labels, train_ratio=0.75, val_ratio=0.15, test_ratio=0.1):
    """
    Split data per individual and per class to avoid temporal overlap.

    Args:
        data: data organized per individual [patient1_data, patient2_data, ...]
        labels: labels organized per individual [patient1_labels, patient2_labels, ...]
        train_ratio, val_ratio, test_ratio: ratios for train/val/test

    Returns:
        train_data, val_data, test_data: split datasets
        train_labels, val_labels, test_labels: split labels
    """
    print("[Data Split] Starting individual and class-based data splitting...")

    # Count positive/negative per individual
    patient_stats = []
    for i, (patient_data, patient_labels) in enumerate(zip(data, labels)):
        pos_count = np.sum(patient_labels == 1)
        neg_count = np.sum(patient_labels == 0)
        patient_stats.append({
            'patient_id': i,
            'pos_count': pos_count,
            'neg_count': neg_count,
            'total': len(patient_labels)
        })
        print(f"Patient {i}: Pos={pos_count}, Neg={neg_count}, Total={len(patient_labels)}")

    # Split per individual
    train_data, val_data, test_data = [], [], []
    train_labels, val_labels, test_labels = [], [], []

    for i, (patient_data, patient_labels) in enumerate(zip(data, labels)):
        # Separate positive/negative indices
        pos_indices = np.where(patient_labels == 1)[0]
        neg_indices = np.where(patient_labels == 0)[0]

        # Compute split sizes per class
        n_pos = len(pos_indices)
        n_neg = len(neg_indices)

        n_train_pos = int(n_pos * train_ratio)
        n_val_pos = int(n_pos * val_ratio)
        n_test_pos = n_pos - n_train_pos - n_val_pos

        n_train_neg = int(n_neg * train_ratio)
        n_val_neg = int(n_neg * val_ratio)
        n_test_neg = n_neg - n_train_neg - n_val_neg

        # Shuffle indices
        np.random.shuffle(pos_indices)
        np.random.shuffle(neg_indices)

        # Positive splits
        train_pos_idx = pos_indices[:n_train_pos]
        val_pos_idx = pos_indices[n_train_pos:n_train_pos + n_val_pos]
        test_pos_idx = pos_indices[n_train_pos + n_val_pos:]

        # Negative splits
        train_neg_idx = neg_indices[:n_train_neg]
        val_neg_idx = neg_indices[n_train_neg:n_train_neg + n_val_neg]
        test_neg_idx = neg_indices[n_train_neg + n_val_neg:]

        # Merge and append to splits
        # Train set
        train_idx = np.concatenate([train_pos_idx, train_neg_idx])
        np.random.shuffle(train_idx)  # shuffle order
        train_data.append(patient_data[train_idx])
        train_labels.append(patient_labels[train_idx])

        # Validation set
        val_idx = np.concatenate([val_pos_idx, val_neg_idx])
        np.random.shuffle(val_idx)
        val_data.append(patient_data[val_idx])
        val_labels.append(patient_labels[val_idx])

        # Test set
        test_idx = np.concatenate([test_pos_idx, test_neg_idx])
        np.random.shuffle(test_idx)
        test_data.append(patient_data[test_idx])
        test_labels.append(patient_labels[test_idx])

    # Convert to numpy arrays
    train_data = np.array(train_data, dtype=object)
    val_data = np.array(val_data, dtype=object)
    test_data = np.array(test_data, dtype=object)
    train_labels = np.array(train_labels, dtype=object)
    val_labels = np.array(val_labels, dtype=object)
    test_labels = np.array(test_labels, dtype=object)

    print(f"[Data Split] Split completed:")
    print(f"  Train: {len(train_data)} patients")
    print(f"  Val: {len(val_data)} patients")
    print(f"  Test: {len(test_data)} patients")

    return train_data, val_data, test_data, train_labels, val_labels, test_labels


# Xavier initialization helper
def init_weights_xavier(module):
    if isinstance(module, (nn.Conv1d, nn.Linear)):
        if module.weight is not None:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


data_root_path = '/root/MetaVA'
save_model_dir = '/root/MetaVA/trained_modelsNet'


# Mirror stdout/stderr to both console and file
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


def MAMLtrain(model, train_data, train_labels, val_data, val_labels, iterations, writer=None):
    train_data, train_labels = dp.pair_shuffle(data=train_data, label=train_labels)
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

                # Initialization
                # PyTorch 2.6: default weights_only=True may fail for full model pickles
                raw_model = torch.load(save_model_dir + '/raw.pkl', weights_only=False)
                meta_learner = MAML.MetaLearner(raw_model)

                model_cl = MAML.MAML(
                    meta_learner,
                    train_data,
                    train_labels,
                    val_data=val_data,
                    val_labels=val_labels,
                    innerlr=innerlr,
                    outerlr=outerlr,
                    update=update,
                )

                device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
                model_cl.to(device)

                optimizer = optim.Adam(model_cl.parameters(), lr=outerlr, weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

                for task_index in tqdm(range(iterations), desc='Task', leave=False):
                    # Pass current task_index to metatrain
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

                    global_step += 1

                    # # Save model every fixed number of tasks (example)
                    # if (task_index + 1) % 2 == 0:
                    #     save_path = os.path.join(data_root_path, 'trained_models', f'metalearning_task_{task_index+1}.pkl')
                    #     torch.save(model_cl, save_path)
                    #     print(f"Model saved at task {task_index+1}, path: {save_path}")

                # Validate after training
                auc, acc = model_cl.metavalid()
                print(f"[MAML] Validation => AUC: {auc:.4f}, Acc: {acc:.4f}")

                if (auc + acc > best_acc + best_auc):
                    best_acc = acc
                    best_auc = auc
                    torch.save(model_cl, save_model_dir + '/metalearning.pkl')

    return


if __name__ == '__main__':
    # Create Tee instance to write stdout/stderr to both console and file
    log_file = open("training_log.txt", "w")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    # 1) Initialize TensorBoard SummaryWriter
    exp_name = "exp{}".format(datetime.now().strftime("%Y%m%d-%H%M%S"))

    # 2) Load data and perform improved split
    print("[Main] Loading raw data...")
    raw_data = np.load(data_root_path + '/data/train_data.npy', allow_pickle=True)
    raw_labels = np.load(data_root_path + '/data/train_label.npy', allow_pickle=True)

    print(f"[Main] Raw data loaded: {len(raw_data)} patients")

    # Use improved split strategy
    train_data, val_data, test_data, train_labels, val_labels, test_labels = split_data_by_individual_and_class(
        raw_data, raw_labels, train_ratio=0.75, val_ratio=0.15, test_ratio=0.1
    )

    # Save split datasets for later use
    split_data_dir = os.path.join(data_root_path, 'split_data')
    if not os.path.exists(split_data_dir):
        os.makedirs(split_data_dir)

    np.save(os.path.join(split_data_dir, 'train_data.npy'), train_data)
    np.save(os.path.join(split_data_dir, 'train_labels.npy'), train_labels)
    np.save(os.path.join(split_data_dir, 'val_data.npy'), val_data)
    np.save(os.path.join(split_data_dir, 'val_labels.npy'), val_labels)
    np.save(os.path.join(split_data_dir, 'test_data.npy'), test_data)
    np.save(os.path.join(split_data_dir, 'test_labels.npy'), test_labels)

    print(f"[Main] Split data saved to {split_data_dir}")

    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    # 3) Initialize base model and save
    model = ecgTransForm(
        input_channels=1,
        mid_channels=64,
        trans_dim=16,
        num_heads=4,
        dropout=0.5,
        num_classes=2,
        stride=2
    ).cuda(0)

    # Xavier initialize weights to get θ0
    model.apply(init_weights_xavier)

    torch.save(model, save_model_dir + '/raw.pkl')

    # 4) Create a model for training
    model_test = ecgTransForm(
        input_channels=1,
        mid_channels=64,
        trans_dim=16,
        num_heads=4,
        dropout=0.5,
        num_classes=2,
        stride=2
    ).cuda(0)

    # Initialize training model weights to match θ0 (then load weights for consistency)
    # model_test.apply(init_weights_xavier)

    model_test.load_state_dict(model.state_dict())

    # 5) MAML training and log to TensorBoard
    MAMLtrain(model=model_test, train_data=train_data, train_labels=train_labels, val_data=val_data,
              val_labels=val_labels, iterations=300)

    print('save success')

    # 6) Close writer and log file after training
    log_file.close()
