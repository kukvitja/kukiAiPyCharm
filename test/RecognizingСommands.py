import json
import random
from Include import Sound
from Include.work_json import remember_data, get_remember_data
from Include.Wikipendia import wiki_search

level_truth = 1
pach_file_train_dataset = "data/traindataset.json"

pach_file_remember = "data/memory.json"

def remember_data_write(pach_file_remember = pach_file_remember, **kwargs):
    del kwargs['arr_text_input'][0]
    str = kwargs['arr_text_input'][0] + ' ' + kwargs['arr_text_input'][1] + ' ' + kwargs['arr_text_input'][2]
    remember_data(str, kwargs['arr_text_input'], pach_file_remember)


def get_remember(pach_file_remember = pach_file_remember, **kwargs):
    del kwargs['arr_text_input'][0]
    task = ' '.join(kwargs['arr_text_input'])
    return get_remember_data(task, pach_file_remember)

# def define_phrase(task):
#     print(task)
#     with open("data/talkdata.json", encoding='utf-8') as talkdata:
#         f = talkdata.read()
#         f = json.loads(f)
#         f = f['dataset']
#
#         try:
#             cleanedSpl = task.split()
#             w = {}
#             arr_word_w = []
#             for i in f:
#                 w[i['name']] = 0
#                 request_text = i['requests']
#                 # print('request_text',request_text)
#
#                 for text in request_text:
#                     procent = (len(arr_word_w)/100)*100
#
#                     # print('procent ', procent)
#                     # print('text ',text)
#                     for key in cleanedSpl:
#                         if key in text and key not in arr_word_w:
#                             arr_word_w.append(key)
#
#
#                             # обєднувати одинакові слова
#                             # формула вісов придумати!!!!!!!!!!!!!!!!!
#                             w[i['name']] = w[i['name']] + procent
#                             # print('key ',key)
#                             # if key in request_text:
#
#             print(arr_word_w)
#             print(w)
#             max_value = max (w.values ())
#             if max_value > level_truth:
#                 return max(w, key=w.get)
#             else:
#                 print("Записать в тренеровочний файл")
#                 remember_data(task,task,pach_file_train_dataset)
#                 # with open('data/newdataset.txt', 'w', encoding='utf-8') as f:
#                 #     f.write(task)
#                 #     f.close()
#
#
#         except:
#             Sound.write()


def command(key, tasks):
    # print('Command', tasks, "ComandKey", key)
    with open("data/talkdata.json", encoding='utf-8') as talkdata:
        f = talkdata.read()
        f = json.loads(f)
        f = f['dataset']
        try:
            for i in f:
                if i['name'] == key:
                    l = len(i['response'])
                    if l > 0:
                        k = round(random.uniform(0, l-1))
                        Sound.talk(i['response'][k])
                    if i["action"] != "None":
                        rez = eval(i["action"])(arr_text_input = tasks.split(), question_text = i['requests'])
                        print(rez)
                        if rez != None:
                            Sound.talk(rez)


        except:
            TypeError.__name__

