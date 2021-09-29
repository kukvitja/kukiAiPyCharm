import os
import json
import random
import requests
from bs4 import BeautifulSoup




# search_pfrase = 'о+войне'
#
#
# search = 'о+войне'
# param = {'s': search}
# print(param)
#
# r = requests.get('https://zagge.ru/', params=param) #отправляем HTTP запрос и получаем результат
#
# soup = BeautifulSoup(r.text, 'html.parser')
# link = soup.find_all('div', {'class': 'loop-entry-thumbnail'})
# l = len(link)
# if l > 0:
#     k = round(random.uniform(0, l-1))
#     url_fackt = link[k].find('a').get('href')
#     print(url_fackt)
#     rf = requests.get(url_fackt)
#     soup_rez = BeautifulSoup(rf.text, 'html.parser')
#
#     link_rez = soup_rez.find('div', {'class': 'post-content'})
#     print(link_rez.get_text())

url = "https://www.googleapis.com/customsearch/v1"
search = 'рецепти на ужен'
param = {
    'key': 'AIzaSyC9z0aEVdY6gvx9_EP8rmfRhIa6n_QR8FM',
    'cx': '010349152742613606712:5h9izqqtp1s',
    'q': search
}
# print(param)

r = requests.get(url, params=param) #отправляем HTTP запрос и получаем результат
arr_url = r.json()['items']
for i in arr_url:
    print(i['title'])
    print(i['link'])

# soup = BeautifulSoup(r.json(), 'json.parser')
# print(soup)
# link = soup.find_all('div', {'class': 'hlcw0c'})
# print(link)
# l = len(link)
# if l > 0:
#     k = round(random.uniform(0, l-1))
#     url_fackt = link[k].find('a').get('href')
#     print(url_fackt)
#     rf = requests.get(url_fackt)
#     soup_rez = BeautifulSoup(rf.text, 'html.parser')
#
#     link_rez = soup_rez.find('div', {'class': 'post-content'})
#     print(link_rez.get_text())
