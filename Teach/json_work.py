"""
@author: lataf 
@file: json_work.py 
@time: 07.04.2024 10:29
Модуль отвечает за 
UML схема модуля
Сценарий работы модуля:
Тест модуля находится в папке model/tests.
"""

import json
import pandas as pd

# https: // habr.com / ru / articles / 554274 /
# Правила well formed JSON:
# Данные написаны в виде пар «ключ:значение»
# Данные разделены запятыми
# Объект находится внутри фигурных скобок {}
# Массив — внутри квадратных []

file = 'dic_data.json'

with open(file, 'r') as f:
    data = json.load(f)

print('man good = ', data['man']["good"])
print('women good = ', data['women']["good"])

with open('df_data.json', 'r') as f:
    data = json.load(f)

print(data["load"])

df_1 = pd.DataFrame(data['df'])  # https://habr.com/ru/companies/otus/articles/731844/
exploded_df = df_1.explode("data", ignore_index=True)  # Чтобы развернуть столбец data
pd.json_normalize(exploded_df['data'])  # нормализуются данные внутри столбца data
df_2 = pd.concat([exploded_df.drop('data', axis=1),  # Убираем столбец data и заменяем его на
                  pd.json_normalize(exploded_df['data'])], axis=1)  # нормализованные данные столбца data
print(df_2)

input_json = """
[
    {
        "type": "1",
        "name": "name 1"
    },
    {
        "type": "2",
        "name": "name 2"
    },
    {
        "type": "1",
        "name": "name 3"
    }
]"""

# Transform json input to python objects
input_dict = json.loads(input_json)

# Filter python objects with list comprehensions
output_dict = [x for x in input_dict if x['type'] == '1' and x['name'] == 'name 3']


# Transform python object back into json
output_json = json.dumps(output_dict)

# Show json
print(output_json)

# df = pd.DataFrame(
#     [[46, 18], [41, 15]],
#     index=["man", "women"],
#     columns=["good", "bad"],
# )
# print(df)

# df.to_json(file, orient='index', indent=4)
# df.to_json(file, orient='index')
#
# df_2 = pd.read_json(file, orient='index')
#
# print(df_2)

# {"man": {"good": 46, "bad": 18}, "women": {"good": 41, "bad": 15}}

# data = {'a_dict': {'a': 0, 'b': 1}, 'a_list': [0, 1, 2, 3]}
# data = {'a_dict': {'a': 0, 'b': 1}, 'a_list': [0, 1, 2, 3],
#         'a_df': {"man": {"good": 46, "bad": 18}, "women": {"good": 41, "bad": 15}}}

# file = 'df_data.json'
# # with open(file, 'w') as f:
# #     json.dump(data, f, indent=4)
