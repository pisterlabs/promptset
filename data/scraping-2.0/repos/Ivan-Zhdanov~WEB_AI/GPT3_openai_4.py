# Создание нового текста от вопроса в OPENAI
import openai
import time
from Wrap_text_simple import wrap_tags3
from GPT3_API_cheker import api_cheker, list_api


# предыдущий ключ с аккаунта ivan.zhdanov.moscow4
# openai.api_key = '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'

# 28/09
# openai.api_key = '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'

# ключ от аккаунта helloword (5$)
# openai.api_key = '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'

model_id = 'gtp-3.5-turbo'
num_text = ()
results = []



def GPT3(query):
    flag = False
    while flag == False:
        # закрыл чтобы посмотреть как идет на официальном api
        # apiorg = api_cheker()
        # api = apiorg[0]
        # org = apiorg[1]
        # print(api)
        # print(org)

        # API levinavi092@gmail.com 14$ добавил 10 к 5    14.92
        api = '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'
        org = 'org-PAxr1I9jpenI9tI6mgGAhi7k'
        openai.api_key = api
        openai.organization = org
        print("Текущий АПИ = ", api)
        try:
            print("КАКОЙ ЗАПРОС ________________________ ", query)

            responce = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k-0613",
                # model="gpt-3.5-turbo",
                # temperature=0,
                # max_tokens=1024,
                max_tokens=2500,
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": f"{query}"},
                ]
            )
            text3 = responce['choices'][0]['message']['content']
            print("************")
            flag = True
            return text3
            break
        except Exception as e:
            print('Название ошибки --', e)
            flag = False
            time.sleep(20)
    return text3


def Chat_converstaion(text2, query_type, i, h2, img):
    if len(text2) > 11000:
        text2 = text2[:11000]

    if query_type == 'text_2_pr':
        query2 = f"Перепиши с дополнениями \n {h2} {text2}"
        query = f'Выдели 3-4 ключевых идей из текста. Добавь новые уточняющие мысли и факты:"""{text2}"""'
        text1 = GPT3(query)
        query2 = f'Напиши в виде простых абзацев. Раскрой пункты более связанно с дополнениями:"""{text1}"""'
        text4 = GPT3(query2)
        # text6 = GPT3(text4)
        print('Исполнение Нейронки', text4)


    if query_type == 'text_1_pr':
        if len(text2) < 200:
            text2 = h2 + '' + text2
        query2 = f'Перепиши с дополнениями:"""{text2}"""'
        text4 = GPT3(query2)
        print(text4)

    if query_type == 'none':
        text4 = text2


    # ТЕКСТ отправляем в функцию обертки тегами
    if img == '':
        img_in = ''
    else:
        img_in = f'<img src="{img}" alt=f"{h2}">'


    num_text = (i, '<h2>' + h2 + '</h2>', wrap_tags3(text4), img_in)
    # ДОБАВЛЕНИЕ В КОРТЕЖ Н2 ТЕКСТА КАРТИНКИ
    results.append(num_text)
    print("Пред-резалт = ", text4)
    return text4

