# -*- coding: utf-8 -*-
import json

a = {
    'мне':'тебе',
    'мою':'твою',
    'мой':'твой',
    'моего':'твоего',
    'тебя': 'меня',
    'моей': 'твоей',
    'я': 'ты'

}

def remember_data(key, text, pach_file):
    data = dict()
    with open(pach_file, encoding='utf-8') as read_file:
        f=read_file.read()
        if len(f) > 0:
            data=json.loads(f)
            data[key] = text
            with open(pach_file, 'w', encoding='utf-8') as wraid_file:
                f = json.dumps(data, ensure_ascii=False)
                wraid_file.write(f)
                read_file.close()
        else:
            data[key] = text
            with open(pach_file, 'w', encoding='utf-8') as wraid_file:
                f = json.dumps(data, ensure_ascii=False)
                wraid_file.write(f)
                wraid_file.close()


def get_remember_data(task, pach_file):
    with open(pach_file, encoding='utf-8') as read_file:
        f = read_file.read()
        data = json.loads(f)
        for key_data in data:
            key_data_list = key_data.split()
            i = 0
            for key_data_worg in key_data_list:
                if key_data_worg in task:
                    # print(key_data_worg)
                    i = i + 1
                    if i > 2:
                        words = data[key_data]
                        words = ' '.join(words)
                        for k in a:
                            if k in words:
                                new_worlds = words.replace(k, a[k])
                                return new_worlds
                        else:
                            return words
        else:
            return 'Я не знаю ты мне не говорил'

        read_file.close()

