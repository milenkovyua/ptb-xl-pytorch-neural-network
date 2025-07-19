from PTBXLModel import PTBXLModel
import torch
import torch.nn as nn
from data_loader import load_model_train_and_test_data

frequence = 100 # For testing purposes we are going to use the 100 Hz data. The 500 Hz data must be used only in the supercomputer
time = 10 # The recorded time of the data in seconds
sequence_features = 12 # how much features from the ECG we are observing
input_nums = frequence * sequence_features * time # this is the amount of data we are going to feed

h1_middle_layer_neurons = int(input_nums / 2) # here we reduce the layers to half the size of the data we use
h2_middle_layer_neurons = int(input_nums / 4) # and for the second hidden layer we use quater the data size

# Instantiate our model
model = PTBXLModel(input_nums, h1_middle_layer_neurons, h2_middle_layer_neurons)
    
# Pick manual seed for randomization
torch.manual_seed(29)

X_train, X_test, y_train, y_test = load_model_train_and_test_data()

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

output_filename = 'ptb-xl-trained-seq100-age35-50.pt'

print(f'Model training finished. Saved trained data to {output_filename}')
torch.save(model.state_dict(), output_filename)


