import re
with open('../data/positive1.txt', encoding='utf-8') as read_file:
    f = read_file.read()
    # print(f.split('\''))
    data = []
    arr = f.split('),(')
    # print(len(arr))
    # print(re.sub(r"[^а-яА-Я]", r' ', arr[100600]))
    for i in arr:
        data.append(re.sub(r"[^а-яА-Я]", r' ', i))

    read_file.close()

    print(data[593])
    # for r in data:
    #     print(r)

# with open('../data/dialog.txt', 'w', encoding='utf-8') as read_file:
#     read_file.write(data.j)
#     read_file.close()