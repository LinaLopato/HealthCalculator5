import streamlit as st

import health_sys as hs

h_s = hs
user = h_s.User()
health = hs.Health()
health.user = user
st.title("Калькулятор здоровья")
with st.sidebar:
    page = st.radio(
        "Подсистема организма",
        ("Жировой запас", "Сердце", "Легкие")
    )
if page == "Жировой запас":
    st.header("""Индекс массы тела (ИМТ)""")
    st.text("Для расчета индекса массы тела введите свой:")
    weight = st.number_input(' вес в килограммах', value=72, placeholder="Вес в килограммах")
    height = st.number_input(' рост в сантиметрах', value=170, placeholder="Рост в см")

    if st.button('Рассчитать функцию желательности '):
        imt = hs.IMT()
        imt.health = health
        imt.load('imt.json')
        h_level, value = imt.calc(weight=weight, height=height)
        st.write(f'Функция желательности ИМТ = {h_level}% ')
        st.write(f' ИМТ = {value}')

        # fig, plt = user.health.create_diagram(['ИМТ', 'Сердце', 'Легкие'], [25, 70, 80])
        # st.pyplot(plt.gcf())
        # plt.close()
        # st.write(f'Калибровочная диаграмма')
        # plt2 = h_s.HarringtonShow()
        # st.pyplot(plt2.gcf())

        # st.pyplot(fig)


elif page == "Сердце":
    st.header("""Сердце:""")
    st.text("Для расчета индекса пульса измерьте свой пульс в покое (ударов в минуту):")
    input_value = st.number_input(' введите свой пульс в поле', value=66,
                                  placeholder="Пульс а покое")
    gender = st.selectbox(' введите свой пол', ('man', 'women'))
    # age = st.number_input(' введите свой возраст', value=35, placeholder="полных лет жизни")
    if st.button('Рассчитать функцию желательности'):
        user.gender = gender
        # user.age = age
        pulse = hs.Pulse()
        pulse.health = health
        pulse.load('pulse.json')
        h_level = pulse.calc(input_value)
        st.write(f'Функция желательности пульса = {h_level}%')


elif page == "Легкие":
    st.header("""Легкие:""")
    st.text("Для расчета индекса легких измерьте задержку дыхания в секундах:")
    input_value = st.number_input(' введите задержку дыхания в секундах в поле', value=55,
                                  placeholder="Задержка дыхания в секундах")
    gender = st.selectbox(' введите свой пол', ('man', 'women'))
    # age = st.number_input(' введите свой возраст', value=35, placeholder="полных лет жизни")
    if st.button('Рассчитать функцию желательности'):
        user.gender = gender
        resp = hs.Resp()
        resp.health = health
        resp.load('resp.json')
        h_level = resp.calc(input_value)
        st.write(f'Функция желательности легких = {h_level}%')

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

    # if __name__ == "__main__":
    #     ...
    # Set prog_name so that the Streamlit server sees the same command line
    # # string whether streamlit is called directly or via `python -m streamlit`.
    # sys.argv = ["streamlit", "run", "health_sys.py", ""]
    # sys.argv = ["streamlit", "run", "main.py"]
    # sys.exit(stcli.main())
