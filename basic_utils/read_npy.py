import numpy as np

# 加载文件
# data = np.load('/Users/shiqissszj/Documents/All_Projects/MetaVa/MetaVA_Codes/MetaVA_V1/data/adapt_data.npy', allow_pickle=True)
data = np.load('/Users/shiqissszj/Documents/All_Projects/MetaVa/MetaVA_Codes/MetaVA_V1/dataHV/adapt_data.npy', allow_pickle=True)
# data = np.load('/data/myj/MetaVA_V1/data/adapt_data.npy', allow_pickle=True)
# labels = np.load('/Users/shiqissszj/Documents/All_Projects/MetaVa/MetaVA_Codes/MetaVA_V1/data/adapt_label.npy', allow_pickle=True)
labels = np.load('/Users/shiqissszj/Documents/All_Projects/MetaVa/MetaVA_Codes/MetaVA_V1/dataHV/adapt_label.npy', allow_pickle=True)
# labels = np.load('/data/myj/MetaVA_V1/data/adapt_label.npy', allow_pickle=True)


# 打印形状和内容信息
print(f"data.shape为: {data.shape}")
print(f"labels.shape为: {labels.shape}")
print(f"data[0].shape为: {data[0].shape}")
print(f"labels[0].shape为: {labels[0].shape}")
cnt = 0
for label in labels:
    u = np.unique(label)
    # print(u)
    if 1 in u:
        cnt += 1
# print(f"data中含有1的样本有 {cnt}")
# print(f"data1.shape为: {data1.shape}")
# print(f"labels1.shape为: {labels1.shape}")
# print(f"data1[0].shape为: {data1[0].shape}")
# print(f"labels1[0].shape为: {labels1[0].shape}")
#
# cnt1 = 0
# for label in labels1:
#     u = np.unique(label)
#     # print(u)
#     if 1 in u:
#         cnt1 += 1
# print(f"data1中含有1的样本有 {cnt1}")


# # 查看前几条数据
# print(f"信号数据: {data[:80]}")
# print(f"标签: {labels[89]}")
# print(f"患者id：{pid[:80]}")

# for i in range(4006):
#     print(data[i].shape)