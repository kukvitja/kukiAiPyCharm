from Include import Sound

from RecognizingСommands import *
import keyboard

# from _thread import start_new_thread


import os

def exit(*args):
    os.startfile(r"C:\Users\Viktor\Documents\Work")


def print_pressed_keys(e):

    if e.name == '`' or e.name == '\'' or e.name == 'ё':
        # start_new_thread()
        Sound.talk('я слушаю')
        tasks = Sound.write()
        command(define_phrase(tasks), tasks)
    elif e.name == 'esc':
        print('You Pressed A Key!')



if __name__ == '__main__':
    keyboard.on_release(print_pressed_keys)
    keyboard.wait()

    # print(exit())`



