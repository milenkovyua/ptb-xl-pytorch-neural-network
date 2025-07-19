import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wfdb
import ast
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
frequence = 100 # For testing purposes we are going to use the 100 Hz data. The 500 Hz data must be used only in the supercomputer
time = 10 # The recorded time of the data in seconds
sequence_features = 12 # how much features from the ECG we are observing
input_nums = frequence * sequence_features * time # this is the amount of data we are going to feed

# testing with 4 middle layers, doesn't produced different results so we are left with only 3 middle layers
# h1_middle_layer_neurons = int(input_nums * 0.75) # here we reduce the layers to half the size of the data we use
# h2_middle_layer_neurons = int(input_nums * 0.5) # and for the second hidden layer we use quater the data size
# h3_middle_layer_neurons = int(input_nums * 0.25) # and for the second hidden layer we use quater the data size
# h4_middle_layer_neurons = int(input_nums * 0.1) # and for the second hidden layer we use quater the data size

h1_middle_layer_neurons = int(input_nums / 2) # here we reduce the layers to half the size of the data we use
h2_middle_layer_neurons = int(input_nums / 4) # and for the second hidden layer we use quater the data size

# age range to filter the patients data for better accuracy and less data manipulations
min_age = 35
max_age = 50

path = '/home/yuli/py-venv-1/ptb-xl-data/'
sampling_rate=frequence

# Create a new model class that inherits nn.Module
class Model(nn.Module):
    # input layer with 12 features of the ptb-xl
    # hidden layer h1 with number of neurons
    # hidden layer h2 with other number of neurons
    # ->> output with 2 classes of sick and healthy
    def __init__(self,
                in_features=input_nums,
                h1=h1_middle_layer_neurons,
                h2=h2_middle_layer_neurons, 
                # h3=h3_middle_layer_neurons,
                # h4=h4_middle_layer_neurons,
                out_features=2):
    # def __init__(self, in_features=12000, h1=6000, h2=3000, out_features=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        # self.fc3 = nn.Linear(h2, h3)
        # self.fc4 = nn.Linear(h3, h4)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        x = self.out(x)

        return x
    
# Pick manual seed for randomization
torch.manual_seed(29)

# Instantiate our model

model = Model()

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

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

# set the criterion/deviation of model to measure the error
# or how far off the predictions are from the data
criterion = nn.CrossEntropyLoss()

# Choose Adam optimizer - lr is learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001 )
# optimizer = torch.optim.Adamax(model.parameters(), lr=0.001)


# Train our model
epochs = 100 # we want to print the result from the 100th epoch as well so we are doing 100 + 1 iterations
loses = []

for i in range(epochs):
    # go forward and get a prediction
    y_pred = model.forward(X_train)

    # measure the loss/error - gonna be high at first
    loss = criterion(y_pred, y_train)

    # keep track of our loses
    loses.append(loss.detach().numpy())

    # print every 10 epochs
    if (i+1) % 10 == 0 or i == 0:
        print(f'Epoch: {i+1} and loss is: {loss:.7f}')

    # Do some backpropagation: take the error rate of forward propagation and feed it back
    # thru the network to fine tune the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'first error value is: {loses[0]:.7f} and last error value is {loses[epochs-1]:.7f}')

print(f'\nTesting results:')

correct = 0
incorrect = 0

with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)

        # will tell us wether or not our network think the patient is sick
        # print(f'{i} : {str(y_val)} - {y_val.argmax().item()}  - {y_test[i]}')

        if (y_val.argmax().item() == y_test[i]):
            correct += 1
        else:
            incorrect += 1

print(f'correct: {correct}, incorrect: {incorrect}, total: {correct+incorrect}')
print(f'Total correct rate: {((correct / (correct+incorrect)) * 100):.2f} %')
print(f'Total incorrect rate: {((incorrect / (correct+incorrect)) * 100):.2f} %')

