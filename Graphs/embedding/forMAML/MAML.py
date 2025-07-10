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
import util
from util import update_module, clone_module
from net1d import MyDataset
from args import maml
import mdataprocess as dp
from sklearn.metrics import roc_auc_score


class BaseLearner(nn.Module):
    def __init__(self, module=None):
        super(BaseLearner, self).__init__()
        self.model = module

    def __getattr__(self, name):
        """代理所有未定义的方法到内部模型"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)  # 转发到内部模型

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)  # 统一使用model属性


def MAML_update(model, lr, grads):
    # ** Description **
    #
    # MAML update on model using gradients(grads) & learning_rate(lr)
    #
    # **

    if grads is not None:
        params = list(model.parameters())
        if not len(list(params)) == len(grads):
            warn = 'WARNING:MAML_update(): Parameters and gradients should have same length, but we get {} & {}'.format(
                len(params), len(grads))
        for p, g in zip(params, grads):
            if g is not None:
                p.data.sub_(lr * g)  # 使用原地减法更新参数
    return update_module(model)


class MetaLearner(BaseLearner):
    # ** Description
    #
    # Inner-loop Learner
    #
    # **

    def __init__(self, model, lr=None, first_order=None):
        super(MetaLearner, self).__init__()
        self.model = model
        self.lr = maml['innerlr'] if lr == None else lr
        self.first_order = maml['first_order'] if first_order == None else first_order

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def adapt(self, loss, first_order=None):
        # ** Description
        #
        # Takes a gradient step on the loss and updates the cloned parameters in place
        #
        # first_order: Default to self.first_order
        #
        # **
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
        # ** Description
        #
        # Returns a MAML-wrapped copy of the module whose parameters and buffers
        # are 'torch.clone'd from the original module
        #
        # **

        if first_order == None:
            first_order = self.first_order

        return MetaLearner(model=clone_module(self.model),
                           lr=self.lr,
                           first_order=first_order
                           )


class MAML(nn.Module):
    def __init__(self, model, data=None, label=None, innerlr=maml['innerlr'], outerlr=maml['outerlr'], update=maml['update'],
                 first_order=maml['first_order'], batch_size=maml['batch_size'], ways=maml['ways'], shots=maml['shots'],
                 B=9, MAX_ITER=50):
        super(MAML, self).__init__()
        self.model = model
        if data is not None and label is not None:
            self.train_x = data[: len(data) * 3 // 4 - 1]
            self.train_y = label[: len(label) * 3 // 4 - 1]
            self.valid_x = data[len(data) * 3 // 4:]
            self.valid_y = label[len(label) * 3 // 4:]
            self.difficulties = np.zeros(len(self.train_y), dtype=np.float32)
            self.times_selected = np.zeros(len(self.train_y), dtype=np.int32)
            self.init_difficulties()
            self.lowest = np.min(self.difficulties)
        else:
            self.train_x = None
            self.train_y = None
            self.valid_x = None
            self.valid_y = None
            self.difficulties = None
            self.times_selected = None
            self.lowest = None

        self.innerlr = innerlr
        self.outerlr = outerlr
        self.update = update
        self.first_order = first_order
        self.batch_size = batch_size
        self.ways = ways
        self.shots = shots
        self.device = torch.device(maml['device'] if torch.cuda.is_available() else 'cpu')
        self.B = B
        self.MAX_ITER = MAX_ITER

    # 其他方法保持不变
    def get_embedding(self, x):
        """提取嵌入特征，委托给内部的 Net1D 模型"""
        return self.model.get_embedding(x)

    def forward(self, *args, **kwargs):
        self.model(*args, **kwargs)

    def calculate_loss(self, data, label):
        """
        计算任务的损失
        :param data: 任务数据
        :param label: 任务标签
        :return: 损失值
        """
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        data = torch.tensor(data, dtype=torch.float32).to(device)
        data = data.unsqueeze(1)
        label = torch.tensor(label, dtype=torch.long).to(device)

        # 假设模型的输出是类别预测
        pred = self.model(data)
        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(pred, label)

        return loss.item()

    def init_difficulties(self):
        """
        对每个任务（这里假设 train_y 的长度=任务数，一人=一个任务）计算难度值 V_b.
        通过训练每个任务并计算损失，使用 Softmax 得到每个任务的难度值。
        """
        Lloss = []
        for idx in range(len(self.train_y)):
            # 这里需要实际训练模型并计算每个任务的损失
            # 假设 train_x 和 train_y 存储每个任务的数据
            train_data = self.train_x[idx]
            train_label = self.train_y[idx]

            # 假设存在一个函数来计算每个任务的损失
            loss = self.calculate_loss(train_data, train_label)  # 请根据需要实现 `calculate_loss`
            Lloss.append(loss)

        Lloss = np.array(Lloss)

        # Softmax'方式计算任务难度值V_b
        expval = np.exp(Lloss - 1.0)
        sumexp = np.sum(expval)
        Vtrain = expval / sumexp
        self.difficulties = Vtrain

        print("[init_difficulties] difficulties=", self.difficulties[:10], "...")
        return

    # --------------- 关键：每次外循环都要调用 "SampleTasks(iteration)" => B 个任务 (Algorithm 2) ---------------
    def SampleTasks(self, iteration):
        """
        修改后的 CL-selector:
        1) 挑选出含有 `1` 类标签的任务，并从中选择难度最小的 20% 任务
        2) 从所有任务中选择剩余的 80% 任务
        3) 使用轮盘赌选择策略，保证任务类别平衡
        """
        # ------------------ 1) 筛选出含有 `1` 类标签的任务 ------------------
        task_with_1 = []
        task_without_1 = []

        for idx in range(len(self.train_y)):
            label = self.train_y[idx]
            if np.sum(label == 1) > 0:  # 如果任务包含 `1` 类标签
                task_with_1.append(idx)
            else:
                task_without_1.append(idx)

        # ------------------ 2) 从含有 `1` 类标签的任务中选择最容易的 20% ------------------
        difficulty_with_1 = self.difficulties[task_with_1]
        sorted_indices_with_1 = np.argsort(difficulty_with_1)  # 按难度排序
        num_1_tasks = int(self.B * 0.3)  # 从含 `1` 类标签的任务中选择 20% 作为元任务

        selected_indices_with_1 = [task_with_1[i] for i in sorted_indices_with_1[:num_1_tasks]]

        # ------------------ 3) 从所有任务中选择剩余的 80% ------------------
        # 选择剩下的任务，排除已选中的任务
        remaining_tasks = list(set(range(len(self.difficulties))) - set(selected_indices_with_1))
        remaining_difficulties = self.difficulties[remaining_tasks]
        sorted_indices_remaining = np.argsort(remaining_difficulties)  # 按难度排序
        num_remaining_tasks = self.B - num_1_tasks  # 剩余 80% 的任务数量

        selected_indices_remaining = [remaining_tasks[i] for i in sorted_indices_remaining[:num_remaining_tasks]]

        # 合并选择的任务
        selected_indices = selected_indices_with_1 + selected_indices_remaining

        # ------------------ 4) 计算每个任务的选择概率 ------------------
        scores = []
        for idx in selected_indices:
            base_score = (1.0 - float(self.times_selected[idx]) / (iteration + 1.0))
            scores.append(base_score)

        # 如果所有的分数都为 0，选择的任务就没有概率，可能需要退回随机选择
        if len(scores) == 0:
            print("[CL-selector] all candidate scores=0, fallback to random selection.")
            scores = [1.0] * self.B

        # 归一化选择概率
        scores = np.array(scores, dtype=np.float32)
        sum_scores = scores.sum()
        if sum_scores <= 1e-9:
            probs = np.ones_like(scores) / len(scores)
        else:
            probs = scores / sum_scores

        # ------------------ 5) 使用轮盘赌选择 B 次任务 ------------------
        selected_indices_with_prob = []
        for _ in range(self.B):
            rand_val = np.random.rand()
            cumulative = 0.0
            for i, p in enumerate(probs):
                cumulative += p
                if rand_val <= cumulative:
                    chosen_task_id = selected_indices[i]
                    selected_indices_with_prob.append(chosen_task_id)
                    self.times_selected[chosen_task_id] += 1
                    break

        # 继续后续步骤，返回选中的任务
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
        新的 metatrain: 一次选 B 个任务，分别做 inner-loop，再累积到 outer-loop 的 loss
        """
        # CL-selector: 一次取 B 个任务
        tasks = self.SampleTasks(iteration=iteration)  # 里面就会返回 B= self.B 个 (train_loader, test_loader)

        loss_func = torch.nn.CrossEntropyLoss()

        sum_loss = torch.tensor(0.0, device=self.device)
        all_probs = []
        all_labels = []

        # 对每个任务做内层更新
        for (dataloader, dataloader_test) in tasks:
            # 克隆模型
            learner = MetaLearner(clone_module(self.model))

            # ------ Inner Loop: 多次更新(例如 self.update 次) ------
            for _ in range(self.update):
                learner.train()
                for batch_idx, batch in enumerate(dataloader):
                    input_x, input_y = tuple(t.to(self.device) for t in batch)

                    pred = learner(input_x)  # shape=(batch_size, n_classes)
                    loss = loss_func(pred, input_y)

                    # 是否需要 / self.batch_size 可根据你 CrossEntropyLoss的reduction而定
                    # 这里假设 CrossEntropyLoss已经是 mean
                    learner.adapt(loss=loss, first_order=self.first_order)

            # ------ Inner Loop结束后，拿 learner 在 Query 上算loss/acc/auc ------
            learner.eval()
            for batch_idx, batch in enumerate(dataloader_test):
                x, y = tuple(t.to(self.device) for t in batch)
                pred = learner(x)
                batch_loss = loss_func(pred, y)

                sum_loss += batch_loss  # 累加

                # 收集概率和 label 用于后面算 overall AUC
                prob = F.softmax(pred, dim=1)[:, 1].detach().cpu().numpy()
                all_probs.extend(prob)

                labels_cpu = y.detach().cpu().numpy()
                all_labels.extend(labels_cpu)

        # 计算总的loss, 这里因为是多任务多batch累加，如果想平均可以再 / (B * num_batch)
        final_loss = sum_loss

        # 计算ACC和AUC
        if len(all_probs) == 0:
            # 没有数据
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

        # 遍历验证集里的每个“任务”
        for index in range(len(self.valid_y)):
            # 调用 valid_per_task 一次
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

        # 如果列表为空，需要做保护，避免除0
        final_auc = sum(auc_list) / len(auc_list) if len(auc_list) > 0 else 0.0
        final_acc = sum(acc_list) / len(acc_list) if len(acc_list) > 0 else 0.0

        return final_auc, final_acc

    def valid_per_task(self, data, label):
        """
        目的：和元训练同样的few-step方式适应：
          1) 从 train_data (Support) 做 maml['valid_step'] 次小步更新
          2) 在 test_data (Query) 上计算 AUC/ACC
        """
        # 1) 克隆模型，避免改到主模型
        learner = copy.deepcopy(self.model)
        learner.to(self.device)

        # 2) 数据处理: FilterNwaysKshots => Support(train), Query(test)
        data = np.expand_dims(data, axis=1)
        train_data, train_label, test_data, test_label = dp.FilterNwaysKshots(
            data=data,
            label=label,
            N=self.ways,
            train_shots=self.batch_size * self.shots,
            test_shots=self.batch_size * self.shots,
            remain=False  # 这里不需要 valid_data，remain=False即可
        )

        # 若 train_data / test_data 为空，直接返回
        if len(train_data) == 0 or len(test_data) == 0:
            print("[valid_per_task] Empty train or test data, skip.")
            return None, None

        print(f"Support (train) shape: {train_data.shape}, Test (query) shape: {test_data.shape}")

        # 3) 构建 Dataloader
        dataset_train = MyDataset(train_data, train_label)
        dataset_test = MyDataset(test_data, test_label)
        trainset = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)
        testset = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False)

        # 4) 定义一个简单的优化器 (Adam)，lr可与元训练的 innerlr 保持一致
        optimizer = optim.Adam(learner.parameters(), lr=5e-3)
        loss_func = torch.nn.CrossEntropyLoss()

        # 5) **Few-step Inner Loop**：只做 maml['valid_step'] 次更新 (而不是多epoch)
        for step in range(maml['valid_step']):
            learner.train()
            for batch_idx, batch in enumerate(trainset):
                input_x, input_y = tuple(t.to(self.device) for t in batch)
                pred = learner(input_x)
                loss = loss_func(pred, input_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 这里不再使用 scheduler/epoch，因为我们只想做少量步

        # 6) 测试阶段(在 Query 上)
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
            pred_prob.extend(probs[:, 1].detach().cpu().numpy())  # 第1列(正类)概率
            # 计算acc
            pred_label = pred.argmax(dim=1)
            correct += (pred_label == input_y).sum().item()
            total_test += input_y.size(0)
            # 收集真正的标签
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
