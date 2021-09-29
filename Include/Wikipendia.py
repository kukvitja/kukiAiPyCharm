import wikipedia
import re




def wiki_search(sentences=5, lg = 'ru', **kwargs):
    text_search = ' '.join(kwargs['arr_text_input'])
    for i in kwargs['question_text']:
        if i in text_search:
            l = len(i) + 1
            search_string = text_search[l:]

    try:
        wikipedia.set_lang(lg)
        text = wikipedia.summary(search_string)
        text = re.sub(r'\([^)]*\)', '', text)
        text = re.sub(r'[=|$|[|]|{|}|!]', r'', text)
        print(text)
        return text
    except:
       return 'Я тебя не поняла'

def all_search (text_search, lg = 'ru'):
    wikipedia.set_lang(lg)
    ny = wikipedia.page(text_search)
    text_content = ny.content
    # print(text_content)
    return text_content

# elif "что такое" in task:
#         search_work = task.split()
#         del search_work[0]
#         del search_work[0]
#         str = ' '.join(search_work)
#         text = wiki_search(str)
#         talk(text)
#         if text != "Я тебя не поняла":
#             talk('Разказать всю информацыю')
#             task = command()
#             if 'да' in task or 'расскажи' in task:
#                 text = all_search(str)
#                 talk('чтобы прервать нажми Esc')
#                 talk(text)
#             else:
#                 return False