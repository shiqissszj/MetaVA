import pandas as pd
import os

# ------- 参数 ---------
infile  = '/Users/shiqissszj/Documents/All_Projects/MetaVa/Logs/search/single_losses.csv'            # 原始文件
outfile = '/Users/shiqissszj/Documents/All_Projects/MetaVa/Logs/search/single_losses_clean.csv'      # 输出文件

# ------- 读取 ---------
df = pd.read_csv(infile)

# 若原文件列名不同，请在此处调整
# 例如 df.columns = ['Epoch','Loss']，则先重命名
if 'Patient' not in df.columns:
    df['Patient'] = pd.NA          # 占位列

# ------- 重新编号逻辑 ---------
new_id = -1            # 起始为 -1，遇到 Epoch==1 时 +1
patient_ids = []

for epoch in df['Epoch']:
    if epoch == 1:
        new_id += 1
    patient_ids.append(new_id)

df['Patient'] = patient_ids

# ------- 可选：排序 & 重置索引 -------
df = df.sort_values(['Patient', 'Epoch']).reset_index(drop=True)

# ------- 保存 ---------
df.to_csv(outfile, index=False)
print(f"Clean file saved to: {outfile}")
