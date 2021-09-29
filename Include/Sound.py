import speech_recognition as sr
import pyttsx3

def write():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)

    # language="uk-UA`

    try:
        task = r.recognize_google(audio, language="ru-RU").lower()
        # print(f"[log] уловил {task}")
    except sr.UnknownValueError:
        print("Я жду команду")
        task = write()
    except sr.RequestError as e:
        print("Ошибка; {0}".format(e))

    return task


def talk(phrase):
    def onWord(name, location, length):
        print('word', name, location, length)

            # break  # finishing the loop
        # keyboard.press('esc')
        # if length > 10:
        #     engine.stop()

    engine = pyttsx3.init()
    engine.setProperty('voice', "Microsoft Irina Desktop - Russian")
    engine.setProperty('rate', 180)
    engine.connect('started-word', onWord)
    engine.say(phrase)
    # engine.say('The quick brown fox jumped over the lazy dog.')
    engine.runAndWait()

