"""
@author: lataf 
@file: health_sys.py
@time: 12.04.2024 13:49
Модуль отвечает за консолидацию модулей
UML схема модуля
Сценарий работы модуля:
Тест модуля находится в папке model/tests.
"""
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_application_parameters():
    # pd.options.mode.chained_assignment = None  # Убирает предупреждение о запутанности
    plt.rcParams["figure.figsize"] = (9, 9)  # Размер графика в дюймах ( ширина, длина)
    plt.rcParams['axes.grid'] = True  # Наличие сетки
    plt.rcParams['figure.facecolor'] = 'white'  # Белый фон графика
    plt.rcParams['lines.linestyle'] = '-'  # Установить стиль линии
    plt.rcParams['lines.linewidth'] = 3  # Установить ширину линии
    pd.set_option('display.max_columns', 20)  # Число показываемых колонок
    pd.set_option('display.width', 200)  # Ширина выводимой таблицы
    pd.set_option('display.float_format', '{:.0f}'.format)  # Показ целых в Датафрейме


set_application_parameters()


def hi():
    return 'Hi'


def str_main():
    import streamlit as st
    # from streamlit.web.cli import main
    # import sys

    # import numpy as np
    # import matplotlib.pyplot as plt

    st.title("Калькулятор здоровья")
    page = st.sidebar.selectbox("Подсистема организма",
                                ["Жировой запас",
                                 "Сердце",
                                 "Легкие"
                                 ])
    st.write(f'Ваш ИМТ = 22')

    if page == "Жировой запас":
        st.header("""Индекс массы тела (ИМТ)""")
        st.text("Для расчета индекса массы тела введите свой:")
        weight = st.number_input(' вес в килограммах', value=72, placeholder="Вес в килограммах")
        height = st.number_input(' рост в сантиметрах', value=170, placeholder="Рост в см")
        # number = st.number_input("Insert a number", value=None, placeholder="Type a number...")

        if st.button('Рассчитать ИМТ'):
            imt = weight / ((height / 100) ** 2)
            imt = round(imt, 2)
            st.write(f'Ваш ИМТ = {imt}')

    elif page == "Сердце":
        st.header("""Сердце:""")
    elif page == "Легкие":
        st.header("""Легкие:""")

        # st.latex(r'''
        #             F(x) = exp(-(\gamma x)^{-1/\gamma}1\{x>0\})
        #             ''')
        # st.text("Для получения результата:")
        # st.markdown(
        #     "* Сгенерируем N нормально распределенных случайных величин $U_i$ [0,1] (среднее и единичная дисперсия).")
        # st.markdown("* Вычислим N  величин с распределением по формуле:")
        # st.latex(r'''
        #                     X_i=\dfrac{1}{\gamma}\left(-lnU_i)^{-\gamma}\right)
        #                 ''')
        # mu, sigma = 0, 1  # mean and standard deviation
        # gamma = st.slider('Желаемая гамма', 0.25, 2.25, 0.5, 0.25)
        # N = st.number_input("Желаемое N", 100, 10000, 10000)
        # U = np.abs(np.random.normal(mu, sigma, N))
        # X = 1 / gamma * (-np.log(U)) ** (-gamma)
        # X2 = X[X < 20]
        # fig, ax = plt.subplots()
        # count, bins, ignored = plt.hist(X2, 100, density=True)
        # plt.plot(bins,
        #          np.exp(- (gamma * bins) ** (-1 / gamma)) * (1 / gamma) * (gamma * bins) ** (-1 / gamma - 1) * gamma,
        #          linewidth=2, color='r')
        # st.pyplot(fig)


class User:
    """Вводи и вывод данных пользователем """

    def __init__(self):
        self.age = 33
        self.gender = 'women'
        self.health: Health = Health(self)  # Система здоровья

    def input(self):
        ...

    @staticmethod
    def output(value):
        plt.show()
        print(f'Сердце - {value}')


class Subsys:
    def __init__(self):
        self.harrington = Harrington()
        self.data = None  # Название файла json с загрузочными данными

    def load(self):
        self.harrington.load()


class Health:
    """Управление объектами """

    def __init__(self, _user: User = None):
        self.user = _user  # Ссылка на родителя
        self.pulse = Pulse(self)  # ресурсы сердечно-сосудистой системы
        self.imt = IMT(self)  # индекс массы тела
        self.resp = Resp(self)  # ресурсы легких
        self.harrington = Harrington1(self)  # Односторонний перевод параметра в безразмерную величину
        self.harrington2 = Harrington2(self)  # перевод параметра в безразмерную величину
        self.subsystems: dict[str, Subsys] = dict()

    # @staticmethod
    @staticmethod
    def create_diagram(par: list, res: list):
        """Показываем диаграмму"""
        params = par
        results = res

        theta = np.linspace(start=0, stop=2 * np.pi, num=len(results), endpoint=False)
        theta = np.concatenate((theta, [theta[0]]))
        results = np.append(results, results[0])
        fig, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]},
                                     figsize=(10, 5), facecolor='#f3f3f3')
        a0.axes.get_xaxis().set_visible(False)  # Убираем надписи на осях
        a0.axes.get_yaxis().set_visible(False)
        a1.axes.get_xaxis().set_visible(False)
        a1.axes.get_yaxis().set_visible(False)
        ax = fig.add_subplot(121, projection='polar')
        ax.plot(theta, results, linewidth=4, color="red", marker='o')
        ax.set_thetagrids(range(0, 360, int(360 / len(params))), params)
        plt.yticks(np.arange(0, 110, 10), fontsize=10)
        ax.set(facecolor='#f3f3f3')
        ax.set_theta_offset(np.pi / 2)
        pl = ax.yaxis.get_gridlines()
        for line in pl:
            line.get_path()._interpolation_steps = 5

        text_block = Health.text_block(params, results)

        plt.subplot(1, 2, 2)
        plt.text(0.05, 0.5, text_block, fontsize=14)  # Выводим сводный показатель и отдельные показатели
        # plt.show()
        return fig, plt

    @staticmethod
    def text_block(par, res) -> str:
        health = int(round(math.prod(res) ** (1 / len(res)), 0))  # Обобщенный показатель Харрингтона
        header = f"Всего здоровье {health}%\n"  # Заголовок - обобщенный показатель Харрингтона
        parameters = ''  # Показатели здоровья
        for i in range(len(res) - 1):
            parameters += f'    {i + 1}.  {par[i]} {res[i]}%\n'  # Собираем все показатели в строку
        text_block = f'{header}{parameters}'
        return text_block


class Harrington:
    """Односторонний критерий Харрингтона """

    def __init__(self, _subsys: Subsys = None):
        """Ахназарова С.Л., Кафаров В.В.; Методы оптимизации эксперимента в химической технологии;1985, с.209"""
        self.health = _subsys  # Ссылка на родителя
        self.type = 'one'  # one, max, min

    def load(self):
        ...


class HarringtonOne(Harrington):
    """Односторонний критерий Харрингтона """

    def __init__(self, _subsys: Subsys = None, y_good=1, y_bad=0):
        """Ахназарова С.Л., Кафаров В.В.; Методы оптимизации эксперимента в химической технологии;1985, с.209"""
        super().__init__(_subsys)
        self.health = _subsys  # Ссылка на родителя
        # self.type = 'one'  # one, max, min
        self.d_good = 0.8  # Назначаем "хороший" параметр, обычно d = 0.8
        self.d_bad = 0.2  # Назначаем "плохой" параметр, обычно d = 0.2
        self.h_good = -math.log(math.log(1 / self.d_good))  # Good результат уb_0 + b_1*y_good = h_good (1)
        self.h_bad = -math.log(math.log(1 / self.d_bad))  # Bad результат b_0 + b_1 * y_bad = h_bad (2)
        self.b_0: float = 0  # Первый коэффициент в уравнении Харрингтона
        self.b_1: float = 0  # Второй коэффициент в уравнении Харрингтона
        self.y_good = y_good  # Назначаем "хороший" параметр Y при self.d_good
        self.y_bad = y_bad  # Назначаем "плохой" параметр Y при self.d_bad
        self.load()  # Считаем коэффициенты в уравнении Харрингтона
        self.h_level: float = 0  # Частная функция желательности (d) Харрингтона для параметра y

    def load(self):
        """ Ахназарова с. 207   d = exp [—ехр(— у')]  у’ = bo + b1 * у' """
        self.b_1 = (self.h_good - self.h_bad) / (self.y_good - self.y_bad)  # Считаем b_1 из уравнений (1) и (2)
        self.b_0 = self.h_good - self.b_1 * self.y_good  # Считаем b_0 из уравнений (1) и (2)

    def calc(self, y: float):
        """ Ахназарова с. 207   d = exp [—ехр(— у')]  у’ = bo + b1 * у' """
        self.h_level = math.exp(-math.exp(-(self.b_0 + self.b_1 * y)))  # Считаем d по Ахназаровой с.207
        return self.h_level  # Частная функция желательности d


class HarringtonTwo:
    """Двухсторонний критерий Харрингтона"""

    def __init__(self, _subsys: Subsys = None, y_good=1, y_bad=0):
        """Ахназарова С.Л., Кафаров В.В.; Методы оптимизации эксперимента в химической технологии;1985, с.209"""
        self.health = _subsys  # Ссылка на родителя
        self.type = 'one'  # one, max, min
        self.min_harrington = HarringtonOne()
        self.max_harrington = HarringtonOne()
        self.d_good = 0.8  # Назначаем "хороший" параметр, обычно d = 0.8
        self.d_bad = 0.2  # Назначаем "плохой" параметр, обычно d = 0.2
        self.h_good = -math.log(math.log(1 / self.d_good))  # Good результат уb_0 + b_1*y_good = h_good (1)
        self.h_bad = -math.log(math.log(1 / self.d_bad))  # Bad результат b_0 + b_1 * y_bad = h_bad (2)
        self.b_0: float = 0  # Первый коэффициент в уравнении Харрингтона
        self.b_1: float = 0  # Второй коэффициент в уравнении Харрингтона
        self.y_good = y_good  # Назначаем "хороший" параметр Y при self.d_good
        self.y_bad = y_bad  # Назначаем "плохой" параметр Y при self.d_bad
        self.load()  # Считаем коэффициенты в уравнении Харрингтона
        self.h_level: float = 0  # Частная функция желательности (d) Харрингтона для параметра y

    def data(self):
        with open('imt.json', 'r') as f:
            data = json.load(f)
        self.min_harrington.y_good = data["min"]["good"]
        self.min_harrington.y_bad = data["min"]["bad"]
        self.max_harrington.y_good = data["max"]["good"]
        self.max_harrington.y_bad = data["max"]["bad"]
        imt_range = range(data["range"]["begin"], data["range"]["end"], 1)
        d_range_1 = []
        for y in imt_range:
            if y > data["optimum"]:
                d = self.max_harrington.calc(y)
                self.type = 'max'
            elif y < data["optimum"]:
                d = self.min_harrington.calc(y)
                self.type = 'min'
            else:
                d = data["opt_d"]
                self.type = 'one'
            d_range_1.append(d)
        return d_range_1

    def load(self):
        """ Ахназарова с. 207   d = exp [—ехр(— у')]  у’ = bo + b1 * у' """
        self.b_1 = (self.h_good - self.h_bad) / (self.y_good - self.y_bad)  # Считаем b_1 из уравнений (1) и (2)
        self.b_0 = self.h_good - self.b_1 * self.y_good  # Считаем b_0 из уравнений (1) и (2)

    def calc(self, y: float):
        """ Ахназарова с. 207   d = exp [—ехр(— у')]  у’ = bo + b1 * у' """
        self.h_level = math.exp(-math.exp(-(self.b_0 + self.b_1 * y)))  # Считаем d по Ахназаровой с.207
        return self.h_level  # Частная функция желательности d


class Harrington1:
    """Односторонний критерий Харрингтона """

    def __init__(self, _health: Health = None):
        """Ахназарова С.Л., Кафаров В.В.; Методы оптимизации эксперимента в химической технологии;1985, с.209"""
        self.health = _health  # Ссылка на родителя
        self.h_good = -math.log(math.log(1 / 0.80))  # Хороший результат по Харрингтону b_0 + b_1*y_good = h_good (1)
        self.h_bad = -math.log(math.log(1 / 0.20))  # Плохой результат по Харрингтону b_0 + b_1 * y_bad = h_bad (2)
        self.b_0: float = 0  # Первый коэффициент в уравнении Харрингтона
        self.b_1: float = 0  # Второй коэффициент в уравнении Харрингтона
        self.d: float = 0  # Частная функция желательности Харрингтона для параметра y

    def calc(self, y_good: float, y_bad: float, y: float):
        """ Ахназарова с. 207   d = exp [—ехр(— у')]  у’ = bo + b1 * у' """
        self.b_1 = (self.h_good - self.h_bad) / (y_good - y_bad)  # Считаем b_1 из уравнений (1) и (2)
        self.b_0 = self.h_good - self.b_1 * y_good  # Считаем b_0 из уравнений (1) и (2)
        self.d = math.exp(-math.exp(-(self.b_0 + self.b_1 * y)))  # Считаем d по Ахназаровой с.207
        # print('h_good ', self.h_good)
        # print('h_bad ', self.h_bad)
        # print('b_1 ', self.b_1)
        # print('b_0', self.b_0)
        # print('d', self.d)
        return self.d


class Harrington11:
    """Два односторонних критерия Харрингтона """

    def __init__(self, _health: Health = None):
        """Ахназарова С.Л., Кафаров В.В.; Методы оптимизации эксперимента в химической технологии;1985, с.209"""
        self.health = _health  # Ссылка на родителя
        self.h_good = -math.log(math.log(1 / 0.80))  # Хороший результат по Харрингтону b_0 + b_1*y_good = h_good (1)
        self.y_good = 0  # Назначаем "хороший" параметр d = 0.8
        self.h_bad = -math.log(math.log(1 / 0.20))  # Плохой результат по Харрингтону b_0 + b_1 * y_bad = h_bad (2)
        self.y_bad = 0  # Назначаем "плохой" параметр d = 0.2
        self.b_0: float = 0  # Первый коэффициент в уравнении Харрингтона
        self.b_1: float = 0  # Второй коэффициент в уравнении Харрингтона
        self.d: float = 0  # Частная функция желательности Харрингтона для параметра y

    def calc(self, y_good: float, y_bad: float, y: float):
        """ Ахназарова с. 207   d = exp [—ехр(— у')]  у’ = bo + b1 * у' """
        self.b_1 = (self.h_good - self.h_bad) / (y_good - y_bad)  # Считаем b_1 из уравнений (1) и (2)
        self.b_0 = self.h_good - self.b_1 * y_good  # Считаем b_0 из уравнений (1) и (2)
        self.d = math.exp(-math.exp(-(self.b_0 + self.b_1 * y)))  # Считаем d по Ахназаровой с.207
        # print('h_good ', self.h_good)
        # print('h_bad ', self.h_bad)
        # print('b_1 ', self.b_1)
        # print('b_0', self.b_0)
        # print('d', self.d)
        return self.d


class Harrington2:
    """Двухсторонний критерий Харрингтона"""

    def __init__(self, _health: Health = None):
        """Ахназарова С.Л., Кафаров В.В.; Методы оптимизации эксперимента в химической технологии;1985, с.207"""
        self.health = _health  # Ссылка на родителя
        self.d_good = 0.8  # Хороший результат по Харрингтону
        self.d: float = 0  # Частная функция желательности Харрингтона для параметра y

        self.y_max = 25  # 65
        self.y_min = 18.5  # 10
        self.current_IMT = None
        self.param = 20.5
        self.d_param = 0.75  # 0.6
        self.y1 = None
        self.y = None
        self.n = None

    def calc(self, y_max: float, y_min: float, y: float):
        """ Ахназарова с. 207   d=exp[—(|у'|)**n  у_d = 2y- (y_max + y_min) / (y_max - y_min) ' """
        y_d = (2 * y - (y_max + y_min)) / (y_max - y_min)  # Считаем у_d из уравнения
        z = math.log(math.fabs(y_d))
        n = math.log(math.log(1 / self.d_good) / z)  # Считаем показатель степени
        self.d = math.exp(-math.fabs(y_d) ** n)  # Считаем d по Ахназаровой с.207
        return self.d

    def calc2(self, x: float):  # Написано Матвеем
        self.y1 = (2 * self.param - (self.y_max + self.y_min)) / (self.y_max - self.y_min)
        self.n = (math.log(math.log(1 / self.d_param))) / (math.log(math.fabs(self.y1)))
        self.y = (2 * x - (self.y_max + self.y_min)) / (self.y_max - self.y_min)
        self.d = math.exp(-(math.fabs(self.y) ** self.n))
        return self.d


class IMT(Subsys):
    """Управление объектами """

    def __init__(self, _health: Health = None):
        super().__init__()
        self.health = _health  # Ссылка на родителя


class Resp(Subsys):
    """Управление объектами """

    def __init__(self, contr: Health = None):
        super().__init__()
        self.controller = contr  # Ссылка на родителя


class Heart(Subsys):
    """Загрузка данных и расчет показателя пульса по Харрингтону """

    def __init__(self, _health: Health = None):
        super().__init__()
        self.health = _health  # Ссылка на родителя
        self.good_pulse = None  # Хороший пульс
        self.bad_pulse = None  # Плохой пульс
        self.current_pulse = None  # Текущий пульс
        self.d_pulse = None  # Показатель Харрингтона
        xls_file = pd.ExcelFile(r' heart.xlsx')  # Импорт excel файла
        self.df = xls_file.parse('Лист1')  # Создание DataFrame

    def pulse(self, gender: str = 'women', age: int = 26, pulse: int = 66):
        df = self.df
        self.good_pulse = int(df.loc[(df['gender'] == gender) & (df['age'] >= age)]['good_pulse'].iloc[0])
        # Фильтруем по полу, возрасту и выводим первый [0] элемент серии значений как целое число
        self.bad_pulse = int(df.loc[(df['gender'] == gender) & (df['age'] >= age)]['bad_pulse'].iloc[0])
        self.current_pulse = pulse
        self.d_pulse = self.health.harrington.calc(self.good_pulse, self.bad_pulse, self.current_pulse)
        print(f' gender\t{gender},\tage\t{age},\tpulse\t{pulse},\td_pulse\t{int(self.d_pulse * 100)}%')


class Pulse(Subsys):
    """Загрузка данных и расчет показателя пульса по Харрингтону """

    def __init__(self, _health: Health = None):
        super().__init__()
        self.health = _health  # Ссылка на родителя
        self.harrington = HarringtonOne()
        self.good_pulse = None  # Хороший пульс
        self.bad_pulse = None  # Плохой пульс
        self.current_pulse = None  # Текущий пульс
        self.d_pulse = None  # Показатель Харрингтона
        # self.load()

    def load(self):
        with open('pulse.json', 'r') as f:
            data = json.load(f)
        if self.health.user.gender == 'man':
            self.harrington.y_good = data["man"]["good"]
            self.harrington.y_bad = data["man"]["bad"]
        else:
            self.harrington.y_good = data["women"]["good"]
            self.harrington.y_bad = data["women"]["bad"]
        self.harrington.load()

    def calc(self, pulse: int = 66):
        self.current_pulse = pulse
        self.d_pulse = self.harrington.calc(self.current_pulse)
        return int(self.d_pulse * 100)
        # print(f'd_pulse\t{int(self.d_pulse * 100)}%')


def HarringtonShow():
    """Просмотр """
    with open('imt.json', 'r') as f:
        data = json.load(f)

    imt_range = range(data["range"]["begin"], data["range"]["end"], 1)
    har_1 = Harrington1()
    d_range_1 = []
    for y in imt_range:
        if y > data["optimum"]:
            d = har_1.calc(data["max"]["good"], data["max"]["bad"], y)
        elif y < data["optimum"]:
            d = har_1.calc(data["min"]["good"], data["min"]["bad"], y)
        else:
            d = data["opt_d"]
        d_range_1.append(d * 100)

    # imt_range = range(10, 65, 1)
    # # imt_range = range(5, 81, 1)
    # har_1 = Harrington1()
    # y_bad_min = 10
    # y_good_min = 15
    # y_bad_max = 64
    # y_good_max = 45
    # y_optimum = 21
    # d_range_1 = []
    # for y in imt_range:
    #     if y > y_optimum:
    #         d = har_1.calc(y_good_max, y_bad_max, y)
    #     elif y < y_optimum:
    #         d = har_1.calc(y_good_min, y_bad_min, y)
    #     else:
    #         d = 0.98
    #     d_range_1.append(d)

    # print(d_range)
    plt.plot(imt_range, d_range_1, label="two one side", marker="o", ms=6, mfc='w')
    # plt.grid()
    plt.title(f'Harrington1 and Harrington2')
    plt.ylabel('health, %', loc='top', fontsize=12)  # fontweight="bold"
    plt.xlabel('imt', loc='right', fontsize=12)
    plt.legend(loc='best')
    return plt
    # plt.show()

    # har_2 = Harrington2()
    # d_range_2 = []
    # for y in imt_range:
    #     d_range_2.append(har_2.calc2(y))
    # plt.plot(imt_range, d_range_2, label="two side", marker="o", ms=6, mfc='w')
    # # plt.grid()
    # plt.title(f'Harrington1 and Harrington2')
    # plt.ylabel('health part', loc='top', fontsize=12)  # fontweight="bold"
    # plt.xlabel('imt', loc='right', fontsize=12)
    # plt.legend(loc='best')
    # plt.show()


if __name__ == "__main__":
    set_application_parameters()
    # user = User()
    # fig2 = user.health.create_diagram(['ИМТ', 'Сердце', 'Легкие'], [40, 70, 80])
    # plt.show()

    user_1 = User()  # Создаем объект Пользователь
    user_1.health.pulse.calc(71)
    objects = user_1.health.subsystems
    objects[user_1.health.resp.__class__.__name__] = user_1.health.resp
    objects[user_1.health.pulse.__class__.__name__] = user_1.health.pulse
    print(list(objects.keys()))  # ['Pulse', 'Health']
    print(list(objects.values()))  # [<__main__.Pulse object >, <__main__.Health object>]

    # pulse2 = Pulse()
    # pulse2.calc(69)

    # print('Показатели Харрингтона и диаграмма здоровья отрисованы')

    # user_1.health.create_diagram(['ИМТ', 'Сердце', 'Легкие', 'Разум'], [52, 81, 92, 88])
    # user_1.health.create_diagram(['ИМТ', 'Сердце', 'Легкие'], [52, 81, 92])
    # user_1.health.create_diagram(['ИМТ', 'Сердце',], [52, 81])
    # user_1.health.create_diagram(['ИМТ', ], [52, ])

    # with open('imt.json', 'r') as f:
    #     data = json.load(f)
    #
    # imt_range = range(data["range"]["begin"], data["range"]["end"], 1)
    # har_1 = Harrington1()
    # d_range_1 = []
    # for y in imt_range:
    #     if y > data["optimum"]:
    #         d = har_1.calc(data["max"]["good"], data["max"]["bad"], y)
    #     elif y < data["optimum"]:
    #         d = har_1.calc(data["min"]["good"], data["min"]["bad"], y)
    #     else:
    #         d = data["opt_d"]
    #     d_range_1.append(d)
