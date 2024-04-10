"""
@author: Matvey
@file: imt_calc.py
@time: 11.02.2024 11:06
Модуль отвечает за
UML схема модуля
Сценарий работы модуля:
Тест модуля находится в папке model/tests.
"""

athlete = 'соответствует результату тренированного атлета'
excellent = 'отличный результат'
good = 'хороший результат'
above_av = 'результат выше среднего'
aver = 'результат в пределах нормы'
below_av = 'результат ниже среднего'
poor = 'плохой результат'
very_poor = 'очень плохой результат'


def heart_st():
    age = int(input('Введите Ваш возраст: '))
    gender = input('Введите Ваш пол М или Ж : ')

    # noinspection PyCompatibility
    def rhr(age, gender):
        pulse = int(input('Введите Ваш пульс в состоянии покоя: '))
        if gender == 'М':
            if 17 < age < 26:
                if 48 < pulse < 56:
                    rhr_ = athlete
                elif 55 < pulse < 62:
                    rhr_ = excellent
                elif 61 < pulse < 66:
                    rhr_ = good
                elif 65 < pulse < 70:
                    rhr_ = above_av
                elif 69 < pulse < 74:
                    rhr_ = aver
                elif 73 < pulse < 82:
                    rhr_ = below_av
                elif pulse > 81:
                    rhr_ = poor
            elif 25 < age < 36:
                if 48 < pulse < 55:
                    rhr_ = athlete
                elif 54 < pulse < 62:
                    rhr_ = excellent
                elif 61 < pulse < 66:
                    rhr_ = good
                elif pulse > 65 and pulse < 71:
                    rhr_ = above_av
                elif pulse > 70 and pulse < 75:
                    rhr_ = aver
                elif pulse > 74 and pulse < 82:
                    rhr_ = below_av
                elif pulse > 81:
                    rhr_ = poor
            elif age > 35 and age < 46:
                if pulse > 49 and pulse < 57:
                    rhr_ = athlete
                elif pulse > 56 and pulse < 63:
                    rhr_ = excellent
                elif pulse > 62 and pulse < 67:
                    rhr_ = good
                elif pulse > 66 and pulse < 71:
                    rhr_ = above_av
                elif pulse > 70 and pulse < 76:
                    rhr_ = aver
                elif pulse > 75 and pulse < 83:
                    rhr_ = below_av
                elif pulse > 82:
                    rhr_ = poor
            elif age > 45 and age < 56:
                if pulse > 49 and pulse < 58:
                    rhr_ = athlete
                elif pulse > 57 and pulse < 64:
                    rhr_ = excellent
                elif pulse > 63 and pulse < 68:
                    rhr_ = good
                elif pulse > 67 and pulse < 72:
                    rhr_ = above_av
                elif pulse > 71 and pulse < 77:
                    rhr_ = aver
                elif pulse > 76 and pulse < 84:
                    rhr_ = below_av
                elif pulse > 83:
                    rhr_ = poor
            elif age > 55 and age < 66:
                if pulse > 50 and pulse < 57:
                    rhr_ = athlete
                elif pulse > 56 and pulse < 62:
                    rhr_ = excellent
                elif pulse > 61 and pulse < 68:
                    rhr_ = good
                elif pulse > 67 and pulse < 72:
                    rhr_ = above_av
                elif pulse > 71 and pulse < 76:
                    rhr_ = aver
                elif pulse > 75 and pulse < 82:
                    rhr_ = below_av
                elif pulse > 81:
                    rhr_ = poor
            elif age > 65:
                if pulse > 49 and pulse < 56:
                    rhr_ = athlete
                elif pulse > 55 and pulse < 62:
                    rhr_ = excellent
                elif pulse > 61 and pulse < 66:
                    rhr_ = good
                elif pulse > 65 and pulse < 70:
                    rhr_ = above_av
                elif pulse > 69 and pulse < 74:
                    rhr_ = aver
                elif pulse > 73 and pulse < 80:
                    rhr_ = below_av
                elif pulse > 79:
                    rhr_ = poor
        elif gender == 'Ж':
            if age > 17 and age < 26:
                if pulse > 53 and pulse < 61:
                    rhr_ = athlete
                elif pulse > 60 and pulse < 66:
                    rhr_ = excellent
                elif pulse > 65 and pulse < 70:
                    rhr_ = good
                elif pulse > 69 and pulse > 74:
                    rhr_ = above_av
                elif pulse > 73 and pulse < 79:
                    rhr_ = aver
                elif pulse > 78 and pulse < 85:
                    rhr_ = below_av
                elif pulse > 84:
                    rhr_ = poor
            elif age > 25 and age < 36:
                if pulse > 53 and pulse < 60:
                    rhr_ = athlete
                elif pulse > 59 and pulse < 65:
                    rhr_ = excellent
                elif pulse > 64 and pulse < 69:
                    rhr_ = good
                elif pulse > 68 and pulse < 73:
                    rhr_ = above_av
                elif pulse > 72 and pulse < 77:
                    rhr_ = aver
                elif pulse > 76 and pulse < 83:
                    rhr_ = below_av
                elif pulse > 82:
                    rhr_ = poor
            elif age > 35 and age < 46:
                if pulse > 53 and pulse < 60:
                    rhr_ = athlete
                elif pulse > 59 and pulse < 65:
                    rhr_ = excellent
                elif pulse > 64 and pulse < 70:
                    rhr_ = good
                elif pulse > 69 and pulse < 74:
                    rhr_ = above_av
                elif pulse > 73 and pulse < 79:
                    rhr_ = aver
                elif pulse > 78 and pulse < 85:
                    rhr_ = below_av
                elif pulse > 84:
                    rhr_ = poor
            elif age > 45 and age < 56:
                if pulse > 53 and pulse < 61:
                    rhr_ = athlete
                elif pulse > 60 and pulse < 66:
                    rhr_ = excellent
                elif pulse > 65 and pulse < 70:
                    rhr_ = good
                elif pulse > 69 and pulse < 74:
                    rhr_ = above_av
                elif pulse > 73 and pulse < 78:
                    rhr_ = aver
                elif pulse > 77 and pulse < 84:
                    rhr_ = below_av
                elif pulse > 83:
                    rhr_ = poor
            elif age > 55 and age < 66:
                if pulse > 53 and pulse < 60:
                    rhr_ = athlete
                elif pulse > 59 and pulse < 65:
                    rhr_ = excellent
                elif pulse > 64 and pulse < 69:
                    rhr_ = good
                elif pulse > 68 and pulse < 74:
                    rhr_ = above_av
                elif pulse > 73 and pulse < 78:
                    rhr_ = aver
                elif pulse > 77 and pulse < 84:
                    rhr_ = below_av
                elif pulse > 83:
                    rhr_ = poor
            elif age > 65:
                if pulse > 53 and pulse < 60:
                    rhr_ = athlete
                elif pulse > 59 and pulse < 65:
                    rhr_ = excellent
                elif pulse > 64 and pulse < 69:
                    rhr_ = good
                elif pulse > 68 and pulse < 73:
                    rhr_ = above_av
                elif pulse > 72 and pulse < 77:
                    rhr_ = aver
                elif pulse > 76 and pulse < 85:
                    rhr_ = below_av
                elif pulse > 83:
                    rhr_ = poor
        print(f'Ваш пульс в состоянии покоя: {rhr_}')

    rhr(age, gender)
    input()
    print(
        'Cтеп-тест представляет собой разновидность сердечного стресс-теста для выявления'
        ' и диагностики сердечно-сосудистых заболеваний.\n' +
        'Это также хороший показатель физической подготовки '
        'и способности человека восстанавливаться после интенсивных упражнений путем\n' +
        'проверки скорости восстановления.')

    def step_test(age, gender):
        cont = input("Введите 'Да', если хотите пройти степ-тест: ")
        while cont == 'Да' or cont == 'да':
            print()
            print('Перед началом теста разомнитесь в течение нескольких минут\n' +
                  'Попросите помощника установить метроном на темп 24 шага в минуту\n' +
                  "Попросите помощника дать команду 'вперёд' и запустить секундомер\n" +
                  'Поднимайтесь на скамью или ступеньку высотой 30 см и спускайтесь '
                  'по одной ноге за раз в течение 3 минут с частотой 24 шага в минуту\n' +
                  'По окончании степ-теста запишите измерьте и запишите частоту сердечных сокращений, ударов в минуту')
            input()
            hrate_1 = int(input('Введите Ваш пульс: '))
            if gender == 'Мужской' or gender == 'мужской':
                if age > 17 and age < 26:
                    if hrate_1 < 79:
                        step_test = excellent
                    elif hrate_1 > 78 and hrate_1 < 90:
                        step_test = good
                    elif hrate_1 > 89 and hrate_1 < 100:
                        step_test = above_av
                    elif hrate_1 > 99 and hrate_1 < 106:
                        step_test = aver
                    elif hrate_1 > 105 and hrate_1 < 117:
                        step_test = below_av
                    elif hrate_1 > 116 and hrate_1 < 129:
                        step_test = poor
                    elif hrate_1 > 128:
                        step_test = very_poor
                elif age > 25 and age < 36:
                    if hrate_1 < 81:
                        step_test = excellent
                    elif hrate_1 > 80 and hrate_1 < 90:
                        step_test = good
                    elif hrate_1 > 89 and hrate_1 < 100:
                        step_test = above_av
                    elif hrate_1 > 99 and hrate_1 < 108:
                        step_test = aver
                    elif hrate_1 > 107 and hrate_1 < 118:
                        step_test = below_av
                    elif hrate_1 > 117 and hrate_1 < 129:
                        step_test = poor
                    elif hrate_1 > 128:
                        step_test = very_poor
                elif age > 35 and age < 46:
                    if hrate_1 < 83:
                        step_test = excellent
                    elif hrate_1 > 82 and hrate_1 < 97:
                        step_test = good
                    elif hrate_1 > 96 and hrate_1 < 104:
                        step_test = above_av
                    elif hrate_1 > 103 and hrate_1 < 113:
                        step_test = aver
                    elif hrate_1 > 112 and hrate_1 < 120:
                        step_test = below_av
                    elif hrate_1 > 119 and hrate_1 < 131:
                        step_test = poor
                    elif hrate_1 > 130:
                        step_test = very_poor
                elif age > 45 and age < 56:
                    if hrate_1 < 87:
                        step_test = excellent
                    elif hrate_1 > 86 and hrate_1 < 98:
                        step_test = good
                    elif hrate_1 > 97 and hrate_1 < 106:
                        step_test = above_av
                    elif hrate_1 > 105 and hrate_1 < 117:
                        step_test = aver
                    elif hrate_1 > 116 and hrate_1 < 123:
                        step_test = below_av
                    elif hrate_1 > 122 and hrate_1 < 133:
                        step_test = poor
                    elif hrate_1 > 132:
                        step_test = very_poor
                elif age > 55 and age < 66:
                    if hrate_1 < 86:
                        step_test = excellent
                    elif hrate_1 > 85 and hrate_1 < 98:
                        step_test = good
                    elif hrate_1 > 97 and hrate_1 < 104:
                        step_test = above_av
                    elif hrate_1 > 103 and hrate_1 < 113:
                        step_test = aver
                    elif hrate_1 > 112 and hrate_1 < 121:
                        step_test = below_av
                    elif hrate_1 > 120 and hrate_1 < 130:
                        step_test = poor
                    elif hrate_1 > 129:
                        step_test = very_poor
                elif age > 65:
                    if hrate_1 < 88:
                        step_test = excellent
                    elif hrate_1 > 87 and hrate_1 < 97:
                        step_test = good
                    elif hrate_1 > 96 and hrate_1 < 104:
                        step_test = above_av
                    elif hrate_1 > 103 and hrate_1 < 114:
                        step_test = aver
                    elif hrate_1 > 113 and hrate_1 < 121:
                        step_test = below_av
                    elif hrate_1 > 120 and hrate_1 < 131:
                        step_test = poor
                    elif hrate_1 > 130:
                        step_test = very_poor
            elif gender == 'Женский' or gender == 'женский':
                if age > 17 and age < 26:
                    if hrate_1 < 85:
                        step_test = excellent
                    elif hrate_1 > 84 and hrate_1 < 99:
                        step_test = good
                    elif hrate_1 > 98 and hrate_1 < 109:
                        step_test = above_av
                    elif hrate_1 > 108 and hrate_1 < 118:
                        step_test = aver
                    elif hrate_1 > 117 and hrate_1 < 127:
                        step_test = below_av
                    elif hrate_1 > 126 and hrate_1 < 141:
                        step_test = poor
                    elif hrate_1 > 140:
                        step_test = very_poor
                elif age > 25 and age < 36:
                    if hrate_1 < 88:
                        step_test = excellent
                    elif hrate_1 > 87 and hrate_1 < 100:
                        step_test = good
                    elif hrate_1 > 99 and hrate_1 < 112:
                        step_test = above_av
                    elif hrate_1 > 111 and hrate_1 < 120:
                        step_test = aver
                    elif hrate_1 > 119 and hrate_1 < 127:
                        step_test = below_av
                    elif hrate_1 > 126 and hrate_1 < 139:
                        step_test = poor
                    elif hrate_1 > 138:
                        step_test = very_poor
                elif age > 35 and age < 46:
                    if hrate_1 < 90:
                        step_test = excellent
                    elif hrate_1 > 89 and hrate_1 < 103:
                        step_test = good
                    elif hrate_1 > 102 and hrate_1 < 111:
                        step_test = above_av
                    elif hrate_1 > 110 and hrate_1 < 119:
                        step_test = aver
                    elif hrate_1 > 118 and hrate_1 < 129:
                        step_test = below_av
                    elif hrate_1 > 128 and hrate_1 < 141:
                        step_test = poor
                    elif hrate_1 > 140:
                        step_test = very_poor
                elif age > 45 and age < 56:
                    if hrate_1 < 94:
                        step_test = excellent
                    elif hrate_1 > 93 and hrate_1 < 105:
                        step_test = good
                    elif hrate_1 > 104 and hrate_1 < 116:
                        step_test = above_av
                    elif hrate_1 > 115 and hrate_1 < 121:
                        step_test = aver
                    elif hrate_1 > 120 and hrate_1 < 130:
                        step_test = below_av
                    elif hrate_1 > 129 and hrate_1 < 136:
                        step_test = poor
                    elif hrate_1 > 135:
                        step_test = very_poor
                elif age > 55 and age < 66:
                    if hrate_1 < 95:
                        step_test = excellent
                    elif hrate_1 > 94 and hrate_1 < 105:
                        step_test = good
                    elif hrate_1 > 104 and hrate_1 < 113:
                        step_test = above_av
                    elif hrate_1 > 112 and hrate_1 < 119:
                        step_test = aver
                    elif hrate_1 > 118 and hrate_1 < 129:
                        step_test = below_av
                    elif hrate_1 > 128 and hrate_1 < 140:
                        step_test = poor
                    elif hrate_1 > 139:
                        step_test = very_poor
                elif age > 65:
                    if hrate_1 < 90:
                        step_test = excellent
                    elif hrate_1 > 89 and hrate_1 < 103:
                        step_test = good
                    elif hrate_1 > 102 and hrate_1 < 116:
                        step_test = above_av
                    elif hrate_1 > 115 and hrate_1 < 123:
                        step_test = aver
                    elif hrate_1 > 122 and hrate_1 < 129:
                        step_test = below_av
                    elif hrate_1 > 128 and hrate_1 < 135:
                        step_test = poor
                    elif hrate_1 > 134:
                        step_test = very_poor
            print('Ваш результат степ-теста: {step_test}')
            cont = input('Хотите ли Вы пройти степ-тест ещё раз ?: ')

    step_test(age, gender)


if __name__ == '__main__':
    heart_st()
