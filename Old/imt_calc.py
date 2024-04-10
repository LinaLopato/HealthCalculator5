"""
@author: lina
@file: imt_calc.py
@time: 11.02.2024 11:06
Модуль отвечает за 
UML схема модуля
Сценарий работы модуля:
Тест модуля находится в папке model/tests.
"""


# noinspection PyCompatibility
def imt_calc(height, weight):
    # # спросить рост
    # height = int(input('Нажмите Enter, введите свой рост в сантиметрах и снова нажмите Enter '))
    # # спросить вес
    # weight = int(input(' Нажмите Enter, ведите свой вес в килограммах и нажмите Enter '))
    # рассчитать индекс массы тела
    imt = weight / (height / 100) ** 2
    # показать индекс массы тела с точностью до 1 знака после запятой
    print('Нажмите Enter')
    print(f'Ваш индекс массы тела равен {imt:.1f}')
    # прокомментировать имт
    print('Нажмите Enter')
    if imt > 40:
        print('У вас ожирение третьей степени')
    elif imt > 35:
        print('У Вас ожирение второй степени')
    elif imt > 30:
        print('У Вас ожирение первой степени')
    elif imt > 25:
        print('У Вас избыточная масса тела')
    elif imt > 18.5:
        print('Ваша масса тела в норме')
    elif imt > 16:
        print('У Вас недостаточная масса тела')
    else:
        print('У Вас острый дефицит массы тела')


if __name__ == '__main__':
    # спросить рост
    _height = int(input('Нажмите Enter, введите свой рост в сантиметрах и снова нажмите Enter '))
    # спросить вес
    _weight = int(input(' Нажмите Enter, ведите свой вес в килограммах и нажмите Enter '))
    imt_calc(_height, _weight)
