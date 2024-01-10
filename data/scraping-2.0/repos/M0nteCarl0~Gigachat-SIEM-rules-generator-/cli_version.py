from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models.gigachat import GigaChat

# Авторизация в сервисе GigaChat
chat = GigaChat(credentials=,
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

while(True):
    user_input = input("User: ")
    messages.append(HumanMessage(content=user_input))
    res = chat(messages)
    messages.append(res)
    print("Bot: ", res.content)