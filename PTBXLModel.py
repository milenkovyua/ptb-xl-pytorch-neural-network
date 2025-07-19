import torch.nn as nn
import torch.nn.functional as F
frequence = 100 # For testing purposes we are going to use the 100 Hz data. The 500 Hz data must be used only in the supercomputer
time = 10 # The recorded time of the data in seconds
sequence_features = 12 # how much features from the ECG we are observing
input_nums = frequence * sequence_features * time # this is the amount of data we are going to feed

h1_middle_layer_neurons = int(input_nums / 2) # here we reduce the layers to half the size of the data we use
h2_middle_layer_neurons = int(input_nums / 4) # and for the second hidden layer we use quater the data size

# Create a new model class that inherits nn.Module
class PTBXLModel(nn.Module):
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