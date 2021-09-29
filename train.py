import json
from nltk_fanc import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import NeuralNet

with open('data/talkdata.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

all_words = []
names = []
xy = []

for intent in dataset['dataset']:
    name = intent['name']
    names.append(name)
    for request in intent['requests']:
        w = tokenize(request)
        all_words.extend(w)
        xy.append((w, name))

ignor_words =['?', '.', ',', '!']
all_words = [stem(w) for w in all_words if w not in ignor_words]
all_words = sorted(set(all_words))
names = sorted(set(names))

X_train = []
Y_train = []

for (patern_sentence, name) in xy:
    bag = bag_of_words(patern_sentence, all_words)
    X_train.append(bag)

    label = names.index(name)
    Y_train.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


# Hiper params

batch_size = len(names)# Размер входных данных для одной итерации
hidden_size = len(X_train[0]) # Количество узлов на скрытом слое
output_size = len(names) # == num_classes = 10# Число классов на выходе.
input_size = len(X_train[0]) # Размеры входних даних
learning_rate = 0.001 # Скорость конвергенции
num_epoch = 400 # Количество тренировок всего набора данных

print(input_size, output_size)

dataset = ChatDataset()

train_loager = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

#Loos and Optimizator
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epoch):
    for (words, labels) in train_loager:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # forward
        outputs = model(words)
        loss = criterion(outputs, labels)

#         backward and optimize step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if  (epoch +1) %100 == 0:
        print(f'epoch {epoch + 1}/{num_epoch}, loss = {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')
print(model.state_dict())
# save model

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"names": names
}

FILE = "data.pt"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')

