from collections import Counter
import numpy as np

train_label = np.load('/Users/shiqissszj/Documents/All_Projects/MetaVa/MetaDemo/processed_data_vtac/train_label.npy', allow_pickle=True)
label_counts = Counter(train_label)
print("Label Distribution in Training Data:", label_counts)