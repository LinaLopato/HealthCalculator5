params = ['ИМТ', 'Сердце', 'Легкие']
results = [50, 80, 90]
dictionary = dict(zip(params, results))
lst = []
for key, value in dictionary.items():
    lst.append(f'{key} {value}%')
    # print(f'{key} {value}%\n')
print(lst)


