# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

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

if __name__ == "__main__":
    ...
    # Set prog_name so that the Streamlit server sees the same command line
    # string whether streamlit is called directly or via `python -m streamlit`.
    # sys.argv = ["streamlit", "run", "main.py", ""]
    # sys.exit(main(prog_name="streamlit"))
