import torch
import numpy as np
import wfdb
import ast
import pandas as pd
from sklearn.model_selection import train_test_split

# age range to filter the patients data for better accuracy and less data manipulations
min_age = 35
max_age = 50

path = 'ptb-xl-data/'


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


def aggregate_diagnostic(y_dic: dict):
    tmp = []
    dic_keys = y_dic.keys()
    if 'NORM' in dic_keys and 'SR' in dic_keys:
        tmp.append(0.0)
    else:
        tmp.append(1.0)

    return list(set(tmp))

def aggregate_diagnostic_by_report(report: str):
    tmp = []
    if report == 'sinusrhythmus normales ekg':
        tmp.append(0.0)
    else:
        tmp.append(1.0)

    return list(set(tmp))

def load_model_train_and_test_data(min_age=35, max_age=50, data_frequence=100, path = './ptb-xl-data/'):
    sampling_rate=data_frequence

    # load and convert annotation data
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # filtering the patients by age, and only if the result is validated by human
    # for higher accuracy
    Y = Y[Y['age'].between(min_age, max_age)]
    Y = Y[(Y['validated_by_human'] == True)]
    # Y = Y[(Y['strat_fold'] == 10)]

    # change filter by report text
    # Y = Y[(Y.report == 'sinusrhythmus normales ekg')]


    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, path)

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]


    # Apply diagnostic superclass
    # Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)
    Y['diagnostic_superclass'] = Y.report.apply(aggregate_diagnostic_by_report)

    # Creating two temporarly X and y matrices, we are going to use to apply further data
    # manipulations to ensure only valid data is used for the training process
    # this will give us high level of predictions
    X_tmp = X
    y_tmp = Y.diagnostic_superclass.values

    X_f = []
    y_f = []

    # normalizing the data
    for i in range(len(X_tmp)):
        if (len(y_tmp[i]) > 0):
            tmparr = np.array(X_tmp[i])
            tmparr = np.concatenate(tmparr)
            X_f.append(tmparr)
            y_f.append(y_tmp[i][0])

    X_tmp = X_f
    y_tmp = y_f

    # Split the data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X_tmp, y_tmp, test_size=0.05, random_state=29)
    X_train, X_test, y_train, y_test = train_test_split(X_tmp, y_tmp, test_size=0.1, random_state=29)

    # Convert X features to float tensors
    X_train = torch.FloatTensor(np.asarray(X_train))
    X_test = torch.FloatTensor(np.asarray(X_test))
    # convert the y labels to tensors long
    # y_train = torch.LongTensor(np.concatenate(y_train))
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    return X_train, X_test, y_train, y_test


