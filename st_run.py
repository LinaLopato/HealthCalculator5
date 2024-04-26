"""
@author: lataf 
@file: st_run.py
@time: 12.04.2024 17:03
Модуль отвечает за 
UML схема модуля
Сценарий работы модуля:
Тест модуля находится в папке model/tests.
"""

import sys
import streamlit.web.cli as stcli

if __name__ == '__main__':
    """Запуск приложения в браузере"""
    # # Set prog_name so that the Streamlit server sees the same command line
    # # string whether streamlit is called directly or via `python -m streamlit`.
    sys.argv = ["streamlit", "run", "st_main.py"]
    # sys.argv = ["streamlit", "run", "Teach/streamlit_work.py"]
    sys.exit(stcli.main())
