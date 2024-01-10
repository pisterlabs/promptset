import streamlit as st
from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models.gigachat import GigaChat

# Авторизация в сервисе GigaChat
chat = GigaChat(credentials=st.secrets["GIGACHAT_CRED"],
                scope='GIGACHAT_API_PERS',
                verify_ssl_certs=False)

messages = [
    SystemMessage(
        content='Обработай запрос. Все числа до первых букв будут являться {timestamp}.\
        Далее выдели {hostname_of_the_system}. Следующим словом будут являться {daemon_creating_the_log}.\
        Число в квадратных скобках будет {process_ID}. После двоеточия будет {log_message}.\
        Результат верни в таком виде:\
        "<decoder name="{daemon_creating_the_log}-auth"> \
        <parent> {daemon_creating_the_log} </parent> \
        <prematch offset="after_parent"> authentication </prematch> \
        <regex offset="after_parent"> ^(\S+) authentication for user (\S+) from (\S+) via (\S+)$</regex>\
        <order>status, srcuser, srcip, protocol</order>\
        </decoder>"'
    )
]

st.title("SIEM rules generator")

while True:
    user_input = st.text_input("User:")
    if st.button("Send"):
        messages.append(HumanMessage(content=user_input))
        res = chat(messages)
        messages.append(res)
        st.write("Bot:", res.content)