import numpy as np
from tqdm import tqdm
import os
from util import MyDataset
from tran_models import ecgTransForm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dataprocess as dp
import copy


model_src = '.'


# Pre-adaptation hyperparameters
pre_iterations = 5  # repeat several times
beta1_inner = 1e-3  # inner update step size β1
beta2_outer = 1e-3  # outer update step size β2
pre_batch_size = 8

# Fine-tuning hyperparameters
finetune_lr = 5e-4
finetune_epochs = 300
finetune_batch_size = 8
early_stop_patience = 10
early_stop_min_delta = 1e-3
weight_decay = 1e-4
min_lr = 1e-6


def select_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def split_indices_by_class(labels: np.ndarray, k_per_class: int):
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    k_pos_train = min(k_per_class, len(pos_idx))
    k_neg_train = min(k_per_class, len(neg_idx))
    # Validation set also takes K per class
    k_pos_val = min(k_per_class, max(0, len(pos_idx) - k_pos_train))
    k_neg_val = min(k_per_class, max(0, len(neg_idx) - k_neg_train))

    pos_perm = np.random.permutation(pos_idx)
    neg_perm = np.random.permutation(neg_idx)

    pos_train = pos_perm[:k_pos_train]
    neg_train = neg_perm[:k_neg_train]

    pos_val = pos_perm[k_pos_train:k_pos_train + k_pos_val]
    neg_val = neg_perm[k_neg_train:k_neg_train + k_neg_val]

    train_idx = np.hstack([pos_train, neg_train])
    val_idx = np.hstack([pos_val, neg_val])

    used_mask = np.zeros_like(labels, dtype=bool)
    used_mask[train_idx] = True
    used_mask[val_idx] = True
    test_idx = np.where(~used_mask)[0]

    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)

    return train_idx, val_idx, test_idx


def build_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = MyDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def pre_fine_tune(model: torch.nn.Module,
                  train_x: np.ndarray,
                  train_y: np.ndarray,
                  beta1: float,
                  beta2: float,
                  iterations: int,
                  batch_size: int,
                  device: torch.device):
    """
    Pre-adaptation: perform small inner-then-outer updates for several iterations.
    - Inner: θ'' = θ' - β1 * ∇_{θ'} L_Dt(f_{θ'})
    - Outer: θ0 = θ'' - β2 * ∇_{θ''} L_Dt(f_{θ''})
    Note: both inner and outer losses are computed on D_t as described.
    """
    loss_func = torch.nn.CrossEntropyLoss()

    for _ in range(iterations):
        # Clone current initialization θ0 to θ'
        inner_model = copy.deepcopy(model)
        inner_model.to(device)

        # Inner update: one gradient descent over D_t (average gradients)
        train_loader = build_loader(train_x, train_y, batch_size=batch_size, shuffle=True)
        inner_model.train()
        for p in inner_model.parameters():
            if p.grad is not None:
                p.grad = None

        for batch in train_loader:
            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = inner_model(input_x)
            loss = loss_func(pred, input_y)
            loss.backward()

        with torch.no_grad():
            for p in inner_model.parameters():
                if p.grad is not None:
                    p.data -= beta1 * p.grad.data

        # Outer update: compute loss on D_t with θ'' and update θ0
        for p in inner_model.parameters():
            if p.grad is not None:
                p.grad = None

        # Re-forward and accumulate gradients (average)
        train_loader = build_loader(train_x, train_y, batch_size=batch_size, shuffle=True)
        for batch in train_loader:
            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = inner_model(input_x)
            loss = loss_func(pred, input_y)
            loss.backward()

        # θ0 ← θ'' - β2 * ∇_{θ''} L
        with torch.no_grad():
            for p_model, p_inner in zip(model.parameters(), inner_model.parameters()):
                grad = p_inner.grad if p_inner.grad is not None else None
                if grad is not None:
                    p_model.data = p_inner.data - beta2 * grad.data
                else:
                    p_model.data = p_inner.data.clone()


def fine_tune_with_validation(model: torch.nn.Module,
                              train_x: np.ndarray,
                              train_y: np.ndarray,
                              val_x: np.ndarray,
                              val_y: np.ndarray,
                              lr: float,
                              epochs: int,
                              batch_size: int,
                              patience: int,
                              min_delta: float,
                              device: torch.device):
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                           patience=5, min_lr=min_lr)

    train_loader = build_loader(train_x, train_y, batch_size=batch_size, shuffle=True)
    val_loader = build_loader(val_x, val_y, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_train_loss = 0.0
        for batch in train_loader:
            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            loss = loss_func(pred, input_y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            num_batches = 0
            for batch in val_loader:
                input_x, input_y = tuple(t.to(device) for t in batch)
                pred = model(input_x)
                loss = loss_func(pred, input_y)
                val_loss += loss.item()
                num_batches += 1
            val_loss = val_loss / max(num_batches, 1)

        scheduler.step(val_loss)

        improve = best_val_loss - val_loss
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            
        model.train()
            
        if patience_counter >= patience:
            break
            
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)


def evaluate_on_test(model: torch.nn.Module,
                     test_x: np.ndarray,
                     test_y: np.ndarray,
                     device: torch.device):
    test_loader = build_loader(test_x, test_y, batch_size=1, shuffle=False)
    model.eval()
    pred_probs = []
    labels = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            input_x, input_y = tuple(t.to(device) for t in batch)
            logits = model(input_x)
            prob = F.softmax(logits, dim=1)[:, 1]
            pred_probs.append(prob.squeeze().cpu().numpy())
            pred_class = logits.argmax(dim=1)
            correct += (pred_class == input_y).sum().item()
            labels.extend([int(i) for i in input_y])
            total += input_y.size(0)

    pred_probs_arr = np.array(pred_probs).reshape(-1)
    labels_arr = np.array(labels)

    if len(np.unique(labels_arr)) < 2:
        auc = 1.0
    else:
        auc = dp.roc_curve(pred_probs_arr, labels_arr)
    acc = correct / max(total, 1)
    return auc, acc


def expand_channel_dim(x: np.ndarray) -> np.ndarray:
    # Input (N, L) -> (N, 1, L)
    return np.expand_dims(x, 1)


def _build_base_model() -> torch.nn.Module:
    # Keep architecture consistent with training setup in pre_train.py
    model = ecgTransForm(
        input_channels=1,
        mid_channels=128,
        trans_dim=16,
        num_heads=4,
        dropout=0.5,
        num_classes=2,
        stride=2,
    )
    return model


def _load_state_dict_safely(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    # Load state_dict saved as .pt (weights only). Handle common key prefixes.
    try:
        state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location='cpu')

    # Some checkpoints might wrap weights in a key like 'state_dict'
    if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
        state = state['state_dict']

    # Strip common prefixes like 'module.' or 'model.'
    if isinstance(state, dict):
        cleaned = {}
        for k, v in state.items():
            if k.startswith('module.'):
                cleaned[k[len('module.'):]] = v
            elif k.startswith('model.'):
                cleaned[k[len('model.'):]] = v
            else:
                cleaned[k] = v
        state = cleaned

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Warn] Missing keys when loading: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"[Warn] Unexpected keys when loading: {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")

    model.to(device)


def run_for_subject(model_name: str,
                    model_src_dir: str,
                    subject_x: np.ndarray,
                    subject_y: np.ndarray,
                    device: torch.device):
    # Instantiate base model and load weights from .pt (state_dict)
    model = _build_base_model()
    ckpt_dir = os.path.join(model_src_dir, 'trained_models')
    ckpt_path = os.path.join(ckpt_dir, model_name + '.pt')
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Expected a .pt state_dict file.")
    _load_state_dict_safely(model, ckpt_path, device)

    train_idx, val_idx, test_idx = split_indices_by_class(subject_y, 10)

    # Assemble data
    subject_x = expand_channel_dim(subject_x)
    x_train, y_train = subject_x[train_idx], subject_y[train_idx]
    x_val, y_val = subject_x[val_idx], subject_y[val_idx]
    x_test, y_test = subject_x[test_idx], subject_y[test_idx]

    # Pre-adaptation: inner-outer loop
    pre_fine_tune(model, x_train, y_train, beta1_inner, beta2_outer, pre_iterations, pre_batch_size, device)

    # Fine-tuning with validation-based early stopping
    fine_tune_with_validation(model, x_train, y_train, x_val, y_val,
                              lr=finetune_lr,
                              epochs=finetune_epochs,
                              batch_size=finetune_batch_size,
                              patience=early_stop_patience,
                              min_delta=early_stop_min_delta,
                              device=device)

    # Evaluate on test set only (compute AUC and ACC)
    auc, acc = evaluate_on_test(model, x_test, y_test, device)
    return auc, acc


if __name__ == '__main__':
    # Data paths (keep consistent with original script if needed)
    adapt_data = np.load('./data/adapt_data.npy', allow_pickle=True)
    adapt_label = np.load('./data/adapt_label.npy', allow_pickle=True)

    # Model list
    netmodel = ['metalearning']

    device = select_device()

    for model_type in netmodel:
        subject_aucs = []
        subject_accs = []
        for i in tqdm(range(len(adapt_data)), desc=f'{model_type}_Subjects'):
            auc, acc = run_for_subject(
                model_name=model_type,
                model_src_dir=model_src,
                subject_x=np.array(adapt_data[i]),
                subject_y=np.array(adapt_label[i]),
                device=device
            )
            subject_aucs.append(auc)
            subject_accs.append(acc)

        avg_auc = float(np.mean(subject_aucs)) if len(subject_aucs) > 0 else 0.0
        avg_acc = float(np.mean(subject_accs)) if len(subject_accs) > 0 else 0.0

        # Finally, print average AUC and ACC to console only
        print(f"{model_type} Average AUC: {avg_auc:.6f}, Average ACC: {avg_acc:.6f}")
