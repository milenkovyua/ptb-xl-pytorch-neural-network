from PTBXLModel import PTBXLModel
import torch
from data_loader import load_model_train_and_test_data

model = PTBXLModel()
X_train, X_test, y_train, y_test = load_model_train_and_test_data()

output_filename = 'ptb-xl-trained-seq100-age35-50.pt'

model.load_state_dict(torch.load(output_filename))

correct = 0
incorrect = 0

with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)

        if (y_val.argmax().item() == y_test[i]):
            correct += 1
        else:
            incorrect += 1

total = correct + incorrect
print(f'Predictions: correct: {correct}, incorrect: {incorrect}, total: {total}')
print(f'Total correct rate: {((correct / (total)) * 100):.2f} %')
print(f'Total incorrect rate: {((incorrect / (total)) * 100):.2f} %')