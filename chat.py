import random
import json

import torch

from model import NeuralNet
from nltk_fanc import bag_of_words, tokenize

from Include import Sound
from Include.Wikipendia import wiki_search
from funk import *
#
# level_truth = 1
pach_file_train_dataset = "data/traindataset.json"
#
# pach_file_remember = "data/memory.json"

# from RecognizingСommands import *
import keyboard

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('data/talkdata.json', 'r', encoding='utf-8') as json_data:
    dataset = json.load(json_data)

FILE = "data.pt"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
names = data['names']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

def define_phrase(task):

    sentence = tokenize(task)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)


    output = model(X)
    _, predicted = torch.max(output, dim=1)
    name = names[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    print(prob.item())
    if prob.item() > 0.80:
        for intent in dataset['dataset']:
            if name == intent["name"]:
                if intent['action'] != "None":
                    task = eval(intent["action"])(arr_text_input=task.split(), question_text=intent['requests'])
                    if task != None:
                        Sound.talk(task)
                if len(intent['response']) > 0:
                    Sound.talk(random.choice(intent['response']))
                return True

    else:
        print("Записать в тренеровочний файл")
        remember_data(task,task,pach_file_train_dataset)
        with open('data/newdataset.txt', 'w', encoding='utf-8') as f:
            f.write(task)
            f.close()
        return False

def print_pressed_keys(e):

    if e.name == '`' or e.name == '\'' or e.name == 'ё':
        # start_new_thread()
        Sound.talk('я слушаю')
        tasks = Sound.write()
        # Команди роботи
        if 'напомни' in tasks:
            Sound.talk(get_remember(pach_file_remember = pach_file_remember, arr_text_input=tasks.split()))
        # Розговор на осносе датасет
        elif define_phrase(tasks):
            pass
        # elif 'напомни' in tasks or 'когда' in tasks:
        #     Sound.talk(get_remember(pach_file_remember = pach_file_remember, arr_text_input=tasks.split()))
        # Розговор на основі даних питання відповідь


        # Отговорки коли не знає що відповісти
        else:
            Sound.talk(random.choice(dataset['answers_know']))
    elif e.name == 'esc':
        print('You Pressed A Key!')



if __name__ == '__main__':
    # keyboard.on_release(print_pressed_keys)
    # keyboard.wait()
    while True:
        Sound.talk('я слушаю')
        tasks = Sound.write()

        if 'напомни' in tasks:
            Sound.talk(get_remember(pach_file_remember=pach_file_remember, arr_text_input=tasks.split()))

        elif define_phrase(tasks):
            pass

        else:
            Sound.talk(random.choice(dataset['answers_know']))
