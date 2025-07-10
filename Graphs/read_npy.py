import numpy as np

# 加载文件
data = np.load('/Users/shiqissszj/Documents/All_Projects/MetaVa/MetaDemo_Finished/data/train_data.npy', allow_pickle=True)
labels = np.load('/Users/shiqissszj/Documents/All_Projects/MetaVa/MetaDemo_Finished/data/train_label.npy', allow_pickle=True)

# 打印形状和内容信息
print(f"data.shape为: {data.shape}")
print(f"labels.shape为: {labels.shape}")
print(f"data[0].shape为: {data[0].shape}")
print(f"labels[0].shape为: {labels[0].shape}")
cnt = 0
for label in labels:
    u = np.unique(label)
    if 1 in u:
        print(f"第{cnt}个数据样本含有1")
    cnt += 1