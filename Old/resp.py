best = 'отлично'
good = 'хорошо'
satisfactory = 'удовлетворительно'
bad = 'плохо'


def respiratory():
    gender = input('Введите ваш пол М или Ж: ')

    def proba(gender):
        time = int(input('Введите время задержки воздуха на вдохе в секундах: '))
        if gender == 'М':
            if time < 35:
                resp_probe = bad
            elif 35 <= time <= 45:
                resp_probe = satisfactory
            elif 45 < time <= 60:
                resp_probe = good
            else:
                resp_probe = best
        elif gender == 'Ж':
            if time < 29:
                resp_probe = bad
            elif 29 <= time <= 40:
                resp_probe = satisfactory
            elif 40 < time <= 55:
                resp_probe = good
            else:
                resp_probe = best
        print('Ваш результат:', resp_probe)

    proba(gender)


print('Cделайте 2-3 глубоких вдоха и выдоха, а затем, сделав полный вдох, задержите дыхание')
print('По окончании времени, нажмите на любую кнопку')
input()
respiratory()
