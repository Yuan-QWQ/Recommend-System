import numpy as np
import torch
import pandas as pd
import sklearn


class CriteoDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, rows):
        num_features = ['I' + str(i) for i in range(1, 14)]
        cat_features = ['C' + str(i) for i in range(1, 27)]

        col_names_train = ['label'] + num_features + cat_features
        self.data = pd.read_csv(filepath, nrows=rows, sep='\t', names=col_names_train)

        self.label = self.data['label']
        self.field_dims = np.zeros(39, dtype=np.int32)
        # 补全
        self.data[num_features] = self.data[num_features].fillna(-1)
        self.data[cat_features] = self.data[cat_features].fillna('NULL')
        # 连续型特征离散化
        self.data[num_features] = self.data[num_features].astype(str)
        for col in num_features:
            #    for i in range(len(self.data[col])):
            #        self.data[col][i] = convert_numeric_feature(self.data[col][i])
            self.data[col] = convert_numeric_feature(self.data[col])
        # 自然数编码
        cols = [f for f in self.data.columns if f not in ['label']]
        for col in cols:
            self.data[col] = label_encode(self.data[col], self.data[col])
        # 计算field_dims
        i = 0
        for col in cols:
            self.field_dims[i] = len(self.data[col].value_counts())
            i += 1
        self.field_dims = torch.tensor(self.field_dims)

    def __getitem__(self, item):
        cols = [f for f in self.data.columns if f not in ['label']]
        return torch.tensor(self.data[cols].iloc[item].values), torch.tensor(float(self.data['label'].iloc[item]))

    def __len__(self):
        return len(self.data)


# 连续型特征离散化
def convert_numeric_feature(val: str):
    v = val.astype(float).astype(int)
    v[v <= 2] = (v - 2)
    v[v > 2] = (np.log(v[v > 2]) ** 2).astype(int)
    v.astype(str)
    return v


# def convert_numeric_feature(val: str):
#     if val == '':
#         return 'NULL'
#     v = int(val)
#     if v > 2:
#         return str(int(math.log(v) ** 2))
#     else:
#         return str(v - 2)

# 自然数编码
def label_encode(series, series2):
    unique = list(series.unique())
    return series2.map(dict(zip(
        unique, range(series.nunique())
    )))
