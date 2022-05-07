import numpy as np
from sklearn.model_selection import train_test_split
import csv
import pandas as pd

filepath = '../JOBS/'
data_file_train = filepath + 'jobs_DW_bin.train.npz'
data_file_test = filepath + 'jobs_DW_bin.test.npz'

df_train = np.load(data_file_train)
df_test = np.load(data_file_test)

y_train = df_train['yf']  # factual observation
t_train = df_train['t'].astype(np.float32)  # treatment
e_train = df_train['e']  # randomized trial
x_train = np.squeeze(df_train['x'])  # confounders
y_test = df_test['yf']  # factual observation
t_test = df_test['t'].astype(np.float32)  # treatment
e_test = df_test['e']  # randomized trial
x_test = np.squeeze(df_test['x'])  # confounders
y = np.concatenate([y_train, y_test], 0)
t = np.concatenate([t_train, t_test], 0)
e = np.concatenate([e_train, e_test], 0)
x = np.concatenate([x_train, x_test], 0)

obs = x.shape[0]
ratio = x_train.shape[0]/obs
yte = np.concatenate([y, t, e], 1)

for i in range(100):
    x_train, x_test, yte_train, yte_test = train_test_split(x, yte, train_size=ratio, test_size=1-ratio)
    y_train, t_train, e_train = yte_train[:, 0], yte_train[:, 1], yte_train[:, 2]
    y_test, t_test, e_test = yte_test[:, 0], yte_test[:, 1], yte_test[:, 2]
    data_train = np.concatenate([x_train, np.expand_dims(t_train, 1), np.expand_dims(y_train, 1), np.expand_dims(e_train, 1)], 1)
    data_test = np.concatenate([x_test, np.expand_dims(t_test, 1), np.expand_dims(y_test, 1), np.expand_dims(e_test, 1)], 1)

    info_x = [(f'x{i}') for i in range(x.shape[1])]
    info_tye = ['t', 'yf', 'e']
    info = info_x + info_tye

    df_tr = pd.DataFrame(data_train)  # A is a numpy 2d array
    df_tr.to_csv(f'./JOBS/jobs_train_{i}.csv', header=info, index=False)

    df_te = pd.DataFrame(data_test)  # A is a numpy 2d array
    df_te.to_csv(f'./JOBS/jobs_test_{i}.csv', header=info, index=False)
