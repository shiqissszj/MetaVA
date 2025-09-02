from torch.autograd import grad
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np
import traceback
from tqdm import tqdm
import copy
import time
import util
from util import update_module, clone_module, MyDataset
from args import maml
import dataprocess as dp
from sklearn.metrics import roc_auc_score


class BaseLearner(nn.Module):
    def __init__(self, module=None):
        super(BaseLearner, self).__init__()
        self.model = module

    def __getattr__(self, attr):
        return super(BaseLearner, self).__getattr__(attr)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def MAML_update(model, lr, grads):
    # Description:
    # Perform a MAML update on model using gradients (grads) and learning rate (lr)

    if grads is not None:
        params = list(model.parameters())
        if not len(list(params)) == len(grads):
            warn = 'WARNING:MAML_update(): Parameters and gradients should have same length, but we get {} & {}'.format(
                len(params), len(grads))
        for p, g in zip(params, grads):
            if g is not None:
                p.data.sub_(lr * g)  # in-place subtraction update
    return update_module(model)


class MetaLearner(BaseLearner):
    # Description:
    # Inner-loop learner wrapper

    def __init__(self, model, lr=None, first_order=None):
        super(MetaLearner, self).__init__()
        self.model = model
        self.lr = maml['innerlr'] if lr == None else lr
        self.first_order = maml['first_order'] if first_order == None else first_order

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def adapt(self, loss, first_order=None):
        # Description:
        # Take a gradient step on the loss and update the cloned parameters in place.
        # first_order: Defaults to self.first_order
        if first_order == None:
            first_order = self.first_order
        grads = grad(loss,
                     self.model.parameters(),
                     retain_graph=not first_order,
                     create_graph=not first_order,
                     allow_unused=True
                     )
        # try:
        #
        # except RuntimeError:
        #     traceback.print_exc()
        #     print("MAML_adapt:something wrong with the autograd_backward")

        self.model = MAML_update(self.model, self.lr, grads)

    def clone(self, first_order=None):
        # Description:
        # Return a MAML-wrapped copy of the module whose parameters and buffers
        # are torch.clone()'d from the original module

        if first_order == None:
            first_order = self.first_order

        return MetaLearner(model=clone_module(self.model),
                           lr=self.lr,
                           first_order=first_order
                           )


class MAML(nn.Module):
    def __init__(self, model, train_data, train_labels, val_data=None, val_labels=None, 
                 innerlr=maml['innerlr'], outerlr=maml['outerlr'], update=maml['update'],
                 first_order=maml['first_order'], batch_size=maml['batch_size'], ways=maml['ways'], shots=maml['shots'],
                 B=9, MAX_ITER=50):
        super(MAML, self).__init__()
        self.model = model

        # Use the provided train/validation data instead of internal split
        self.train_x = train_data
        self.train_y = train_labels
        
        if val_data is not None and val_labels is not None:
            # Use separate validation set
            self.valid_x = val_data
            self.valid_y = val_labels
            print(f"[MAML] Using separate validation set: {len(self.valid_x)} patients")
        else:
            # If no validation provided, split from training data (backward compatible)
            self.valid_x = train_data[len(train_data) * 3 // 4:]
            self.valid_y = train_labels[len(train_labels) * 3 // 4:]
            self.train_x = train_data[: len(train_data) * 3 // 4 - 1]
            self.train_y = train_labels[: len(train_labels) * 3 // 4 - 1]
            print(f"[MAML] Using split validation set: {len(self.valid_x)} patients from training data")

        self.innerlr = maml['innerlr']
        self.outerlr = maml['outerlr']
        self.update = maml['update']
        self.first_order = maml['first_order']
        self.batch_size = maml['batch_size']
        self.ways = maml['ways']
        self.shots = maml['shots']

        self.device = torch.device(maml['device'] if torch.cuda.is_available() else 'cpu')
        self.B = B
        self.MAX_ITER = MAX_ITER

        # ---------------- CL-related parameters ----------------
        # 1) Difficulty value per task (per person)
        self.difficulties = np.zeros(len(self.train_y), dtype=np.float32)
        # 2) Times each task has been selected
        self.times_selected = np.zeros(len(self.train_y), dtype=np.int32)

        # Initialize difficulties once (Algorithm 1)
        self.init_difficulties()

        self.lowest = np.min(self.difficulties)

    def forward(self, *args, **kwargs):
        self.model(*args, **kwargs)

    def calculate_loss(self, data, label):
        """
        Compute task loss.
        :param data: task data
        :param label: task labels
        :return: loss value
        """
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        data = torch.tensor(data, dtype=torch.float32).to(device)
        data = data.unsqueeze(1)
        label = torch.tensor(label, dtype=torch.long).to(device)

        pred = self.model(data)
        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(pred, label)

        return loss.item()

    def init_difficulties(self):
        """
        Algorithm 1 — CL-selector initialization:
        1) For each training task Tb: perform a small adaptation using K=shots per class (support set);
        2) Compute average validation loss on the remaining records of the task;
        3) Softmax: V_b = (exp(loss_b)) / sum_i(exp(loss_i)).
        """
        start_time = time.time()
        print("[init_difficulties] Starting proper difficulty computation with K-shot adaptation...")

        Lloss = []
        for idx in range(len(self.train_y)):
            print(f"[init_difficulties] Processing task {idx+1}/{len(self.train_y)}...")
            
            # All samples for this subject (without channel dim)
            data_i = np.array(self.train_x[idx], dtype=object)
            label_i = np.array(self.train_y[idx])
            
            try:
                # Take K per class for support; validation set is the remaining samples
                tr_x, tr_y, te_x, te_y, rem_x, rem_y = dp.FilterNwaysKshots(
                    data=data_i,
                    label=label_i,
                    N=self.ways,
                    train_shots=self.shots,
                    test_shots=0,
                    remain=True
                )
            except Exception as e:
                print(f"[init_difficulties] Task {idx} FilterNwaysKshots failed: {e}, using fallback")
                # Fallback: use full-task loss as approximation
                loss = self.calculate_loss(data_i, label_i)
                Lloss.append(loss)
                continue

            # Clone and perform a small adaptation on the support set
            learner = MetaLearner(clone_module(self.model))
            learner.to(self.device)
            learner.train()
            
            if len(tr_x) > 0:
                # Do one small adaptation on support (inner-loop update)
                try:
                    sx_np = np.asarray(tr_x[0], dtype=np.float32)
                    sy_np = int(tr_y[0]) if isinstance(tr_y[0], (np.integer, int)) else int(np.asarray(tr_y[0]).item())
                    
                    sx_t = torch.tensor(sx_np, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(1)
                    sy_t = torch.tensor([sy_np], dtype=torch.long, device=self.device)
                    
                    pred = learner(sx_t)
                    s_loss = torch.nn.CrossEntropyLoss()(pred, sy_t)
                    learner.adapt(s_loss, first_order=self.first_order)
                except Exception as e:
                    print(f"[init_difficulties] Task {idx} adaptation failed: {e}, using fallback")
                    loss = self.calculate_loss(data_i, label_i)
                    Lloss.append(loss)
                    continue

            # Validation loss: prefer remaining samples; if none, use full task
            if len(rem_x) == 0:
                val_x, val_y = data_i, label_i
            else:
                val_x, val_y = rem_x, rem_y

            # Compute average loss over validation samples
            learner.eval()
            with torch.no_grad():
                if len(val_x) == 0:
                    Lloss.append(0.0)
                else:
                    loss_sum = 0.0
                    count = 0
                    for sx_np, sy_np in zip(val_x, val_y):
                        try:
                            sx_np = np.asarray(sx_np, dtype=np.float32)
                        except Exception:
                            sx_np = np.array(sx_np).astype(np.float32, copy=False)
                        sy_int = int(sy_np) if isinstance(sy_np, (np.integer, int)) else int(np.asarray(sy_np).item())

                        sx_t = torch.tensor(sx_np, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(1)
                        sy_t = torch.tensor([sy_int], dtype=torch.long, device=self.device)
                        pred_val = learner(sx_t)
                        loss_sum += torch.nn.CrossEntropyLoss()(pred_val, sy_t).item()
                        count += 1
                    
                    # Compute average loss: loss_b = loss_b / (|T_b| - K)
                    avg_loss = loss_sum / max(1, count)
                    Lloss.append(avg_loss)

        Lloss = np.array(Lloss)
        
        # Compute task difficulty V_b using Softmax'
        expval = np.exp(Lloss - 1.0)
        sumexp = np.sum(expval)
        Vtrain = expval / sumexp
        self.difficulties = Vtrain

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"[init_difficulties] Proper difficulty computation completed in {elapsed_time:.2f} seconds")

        # Write elapsed time to a dedicated log file
        try:
            with open("difficulty_computation_log.txt", "a") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] init_difficulties (proper): {elapsed_time:.2f}s\n")
        except Exception as e:
            print(f"[Warning] Failed to write to difficulty log: {e}")

        print("[init_difficulties] difficulties=", self.difficulties[:10], "...")
        return

    # --------------- Call "SampleTasks(iteration)" each outer loop => B tasks (Algorithm 2) ---------------
    def SampleTasks(self, iteration):
        """
        Modified CL-selector:
        1) As original: first take a fixed portion of the easiest tasks that contain class '1' (unchanged)
        2) For the remaining tasks: use a dynamic difficulty threshold + selection count penalty + roulette sampling
        """
        # ------------------ 1) Filter tasks containing class '1' ------------------
        task_with_1 = []
        task_without_1 = []

        for idx in range(len(self.train_y)):
            label = self.train_y[idx]
            if np.sum(label == 1) > 0:  # if task contains class '1'
                task_with_1.append(idx)
            else:
                task_without_1.append(idx)

        # ------------------ 2) Choose the easiest 30% from tasks containing class '1' ------------------
        difficulty_with_1 = self.difficulties[task_with_1]
        sorted_indices_with_1 = np.argsort(difficulty_with_1)  # sort by difficulty
        num_1_tasks = int(self.B * 0.3)  # choose 30% from tasks containing class '1'

        selected_indices_with_1 = [task_with_1[i] for i in sorted_indices_with_1[:num_1_tasks]]

        # ------------------ 3) For remaining tasks use dynamic threshold + probabilistic sampling ------------------
        # Select the remaining tasks, excluding the ones already chosen
        remaining_tasks = list(set(range(len(self.difficulties))) - set(selected_indices_with_1))
        num_remaining_tasks = self.B - num_1_tasks

        # Dynamic difficulty threshold: gradually relax from easy to hard
        progress = float(np.exp(iteration - self.MAX_ITER))
        progress = min(1.0, max(0.0, progress))
        thres = float(self.lowest + (1.0 - self.lowest) * progress)

        # Candidate set: remaining tasks below the threshold
        candidate_indices = [idx for idx in remaining_tasks if self.difficulties[idx] < thres]

        # If no candidates, fall back to sorting by difficulty (original logic)
        if len(candidate_indices) == 0:
            remaining_difficulties = self.difficulties[remaining_tasks]
            sorted_indices_remaining = np.argsort(remaining_difficulties)
            selected_indices_remaining = [remaining_tasks[i] for i in sorted_indices_remaining[:num_remaining_tasks]]
        else:
            # Selection probability: score_k = max(0, 1 - t_k / max(1, iteration))
            scores = []
            denom = float(max(1, iteration))
            for idx in candidate_indices:
                base = 1.0 - float(self.times_selected[idx]) / denom
                scores.append(max(0.0, base))
            scores = np.array(scores, dtype=np.float32)
            sum_scores = float(scores.sum())
            if sum_scores <= 1e-9:
                probs = np.ones_like(scores) / float(len(scores))
            else:
                probs = scores / sum_scores

            # Roulette sampling to pick the remaining number
            selected_indices_remaining = []
            for _ in range(num_remaining_tasks):
                r = np.random.rand()
                cumulative = 0.0
                chosen = None
                for i, p in enumerate(probs):
                    cumulative += float(p)
                    if r <= cumulative:
                        chosen = candidate_indices[i]
                        break
                if chosen is None:
                    chosen = candidate_indices[-1]
                selected_indices_remaining.append(chosen)
                self.times_selected[chosen] += 1

        # Combine selected tasks: keep a fixed portion with class '1' + remaining via probabilistic sampling
        selected_indices_with_prob = selected_indices_with_1 + selected_indices_remaining

        # Return selected tasks
        task_loaders = []
        for idx in selected_indices_with_prob:
            data_i = self.train_x[idx]
            label_i = self.train_y[idx]
            label_i = np.array(label_i)

            train_data, train_label, test_data, test_label = dp.FilterNwaysKshots(
                data_i,
                label_i,
                N=self.ways,
                train_shots=self.batch_size * self.shots,
                test_shots=self.batch_size * self.shots
            )
            train_data = np.expand_dims(train_data, 1)
            test_data = np.expand_dims(test_data, 1)

            dataset_train = MyDataset(train_data, train_label)
            dataset_test = MyDataset(test_data, test_label)

            dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size)
            dataloader_test = DataLoader(dataset_test, batch_size=self.batch_size)

            task_loaders.append((dataloader_train, dataloader_test))

        return task_loaders

    def metatrain(self, iteration=0):
        """
        The CL selector samples B tasks at once, performs inner-loop updates, and accumulates outer-loop loss.
        """
        # # Periodically recompute task difficulty based on current model state
        # if iteration > 0 and (iteration % 50 == 0):
        #     print(f"[MAML] Recomputing task difficulties at iteration {iteration} ...")
        #     recompute_start = time.time()
        #     self.init_difficulties()
        #     self.lowest = np.min(self.difficulties)
        #     recompute_end = time.time()
        #     recompute_time = recompute_end - recompute_start
        #     print(f"[MAML] Difficulty recomputation at iteration {iteration} completed in {recompute_time:.2f} seconds")
        #
        #     # Write recomputation time to a dedicated log file
        #     try:
        #         with open("difficulty_computation_log.txt", "a") as f:
        #             f.write(
        #                 f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Recompute at iteration {iteration}: {recompute_time:.2f}s\n")
        #     except Exception as e:
        #         print(f"[Warning] Failed to write to difficulty log: {e}")

        tasks = self.SampleTasks(iteration=iteration)

        loss_func = torch.nn.CrossEntropyLoss()

        sum_loss = torch.tensor(0.0, device=self.device)
        all_probs = []
        all_labels = []

        # For each task, perform inner updates
        for (dataloader, dataloader_test) in tasks:
            learner = MetaLearner(clone_module(self.model))

            # ------ Inner Loop: multiple updates ------
            for _ in range(self.update):
                learner.train()
                for batch_idx, batch in enumerate(dataloader):
                    input_x, input_y = tuple(t.to(self.device) for t in batch)

                    pred = learner(input_x)  # shape=(batch_size, n_classes)
                    loss = loss_func(pred, input_y)

                    learner.adapt(loss=loss, first_order=self.first_order)

            # ------ After inner loop, evaluate learner on Query to get loss/acc/auc ------
            learner.eval()
            for batch_idx, batch in enumerate(dataloader_test):
                x, y = tuple(t.to(self.device) for t in batch)
                pred = learner(x)
                batch_loss = loss_func(pred, y)

                sum_loss += batch_loss  # accumulate

                # Collect probabilities and labels for overall AUC later
                prob = F.softmax(pred, dim=1)[:, 1].detach().cpu().numpy()
                all_probs.extend(prob)

                labels_cpu = y.detach().cpu().numpy()
                all_labels.extend(labels_cpu)

        final_loss = sum_loss

        # Compute ACC and AUC
        if len(all_probs) == 0:
            return final_loss, 0.0, 0.0

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        pred_binary = (all_probs > 0.5).astype(int)
        correct = (pred_binary == all_labels).sum()
        train_acc = correct / len(all_labels)

        try:
            train_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            train_auc = 0.0

        return final_loss, train_acc, train_auc

    def metavalid(self):
        auc_list = []
        acc_list = []
        skipped_tasks = 0

        for index in range(len(self.valid_y)):
            temauc, temacc = self.valid_per_task(
                data=np.array(self.valid_x[index]),
                label=np.array(self.valid_y[index])
            )
            if temauc is not None and not np.isnan(temauc):
                auc_list.append(temauc)
            else:
                skipped_tasks += 1

            if temacc is not None and not np.isnan(temacc):
                acc_list.append(temacc)

        print(f"[MAML] Skipped {skipped_tasks} tasks due to single-class labels or AUC error.")

        # If lists are empty
        final_auc = sum(auc_list) / len(auc_list) if len(auc_list) > 0 else 0.0
        final_acc = sum(acc_list) / len(acc_list) if len(acc_list) > 0 else 0.0

        return final_auc, final_acc

    def valid_per_task(self, data, label):
        """
        Goal: few-step adaptation similar to meta-training:
          1) Perform maml['valid_step'] small updates on train_data (Support)
          2) Compute AUC/ACC on test_data (Query)
        """
        # 1) Clone the model to avoid changing the main model
        learner = copy.deepcopy(self.model)
        learner.to(self.device)

        # 2) Data split: FilterNwaysKshots => Support(train), Query(test)
        data = np.expand_dims(data, axis=1)
        train_data, train_label, test_data, test_label = dp.FilterNwaysKshots(
            data=data,
            label=label,
            N=self.ways,
            train_shots=self.batch_size * self.shots,
            test_shots=self.batch_size * self.shots,
            remain=False  # no need to return remaining data here
        )

        # If train_data/test_data is empty, skip
        if len(train_data) == 0 or len(test_data) == 0:
            print("[valid_per_task] Empty train or test data, skip.")
            return None, None

        print(f"Support (train) shape: {train_data.shape}, Test (query) shape: {test_data.shape}")

        # 3) Dataloader
        dataset_train = MyDataset(train_data, train_label)
        dataset_test = MyDataset(test_data, test_label)
        trainset = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)
        testset = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False)

        # 4) Define a simple optimizer (Adam); lr can match meta-training innerlr
        optimizer = optim.Adam(learner.parameters(), lr=5e-3)
        loss_func = torch.nn.CrossEntropyLoss()

        # 5) Few-step Inner Loop: do exactly maml['valid_step'] updates
        for step in range(maml['valid_step']):
            learner.train()
            for batch_idx, batch in enumerate(trainset):
                input_x, input_y = tuple(t.to(self.device) for t in batch)
                pred = learner(input_x)
                loss = loss_func(pred, input_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # No scheduler/epoch here; only a few steps

        # 6) Test phase (on Query)
        learner.eval()
        pred_prob = []
        acc_label = []
        correct = 0
        total_test = 0

        # with torch.no_grad():
        for batch_idx, bath in enumerate(testset):
            input_x, input_y = tuple(t.to(self.device) for t in batch)
            pred = learner(input_x)  # shape=(B, 2)
            probs = F.softmax(pred, dim=1)
            pred_prob.extend(probs[:, 1].detach().cpu().numpy())

            pred_label = pred.argmax(dim=1)
            correct += (pred_label == input_y).sum().item()
            total_test += input_y.size(0)

            acc_label.extend(l.item() for l in input_y.cpu())

        if total_test == 0:
            print("[valid_per_task] No test data?!")
            return None, None
        acc = correct / total_test

        unique_labels = np.unique(acc_label)
        if len(unique_labels) < 2:
            print("[valid_per_task] Only one class in test set. Skip AUC.")
            return None, acc
        else:
            try:
                AUC = roc_auc_score(acc_label, pred_prob)
            except ValueError as e:
                print(f"AUC Calculation Error: {e}")
                AUC = None

        print(f"[valid_per_task] AUC={AUC}, ACC={acc}, steps={maml['valid_step']}")
        return AUC, acc
