import os
import openai
from dotenv import load_dotenv
from .summarize_utils import summarize_text, evaluate_text_reduction
import time

load_dotenv()
# Инициализация OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')



def get_response(messages):
    '''Получение ответа на ChatGPT'''
    time.sleep(10)
    max_attempts = 3
    current_attempt = 1
    while current_attempt <= max_attempts:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            return response
        except Exception as e:
            print(f"Ошибка запроса: {str(e)}")
            if current_attempt < max_attempts:
                sleep_time = 10 + 10 * current_attempt
                print(f'Повторная попытка через {sleep_time} секунд...')
                time.sleep(sleep_time)
            current_attempt += 1
    return None



def summarize_clusters(clusters):

    # Общее количество токенов на сводку
    summarize_tokens = 0
    # Создаем список для хранения словарей оценок
    evaluation_results = []
    # Создаем словарь для хранения средних значений каждого параметра
    average_scores = {
        "Коэффициент Жаккара": 0.0,
        "Косинусное расстояние": 0.0,
        "BLEU": 0.0,
        "ROUGE F-мера": 0.0,
        "Удельная частота уникальных слов (исходный текст)": 0.0,
        "Удельная частота уникальных слов (сокращенный текст)": 0.0
    }

    # Перебор каждого кластера
    for cluster_label, cluster_dict in clusters.items():
        texts = "\n".join([news_item.text for news_item in cluster_dict['items']])
       
        # Сокращение текстов новостей
        text = summarize_text(texts)
        print(f'\n\nКластер {cluster_label}\n\n', text, '\n')

        messages=[
            {"role": "system", "content": "Ты программа, которой я отправляю несколько текстов новостей на одну тему. Твоя задача - придумывать заголовки и создавать новостной блок из присланного текста. Укажи главные факты и события из текстов."},
            {"role": "user", "content": f'Вот текст новостей: {text}'},
            {"role": "assistant", "content": "Что мне сделать с этим текстом?"},
            {"role": "user", "content": "Придумай заголовок"},
        ]

        # Запрос заголовка
        cluster_title = get_response(messages)
        print('TITLE\n', cluster_title.choices[0].message.content.strip('"'))

        messages.append(cluster_title.choices[0].message)
        messages.append({"role": "user", "content": "Создай новостной блок"})

        # Запрос текста сводки
        cluster_text = get_response(messages)
        print('TEXT\n', cluster_text.choices[0].message.content.replace("\n\n", "\n"))

        # Считаем общее количество потраченных токенов
        summarize_tokens += cluster_title.usage.total_tokens + cluster_text.usage.total_tokens

        # Достаем текст заголовка и новостной сводки из response
        cluster_title = cluster_title.choices[0].message.content.strip('"').strip()
        cluster_text = cluster_text.choices[0].message.content.replace("\n\n", "\n").strip()

        # Сохраняем заголовки и текст
        cluster_dict['cluster_title'] = cluster_title
        cluster_dict['cluster_text'] = cluster_text

        evaluation_results.append(evaluate_text_reduction(texts, cluster_title + ' ' + cluster_text))
              
    # Вычисляем сумму значений каждого параметра из всех оценок
    for evaluation in evaluation_results:
        for key, value in evaluation.items():
            average_scores[key] += value

    # Вычисляем средние значения каждого параметра
    num_evaluations = len(evaluation_results)
    for key in average_scores:
        average_scores[key] /= num_evaluations

    print('Средняя суммаризации:\n', average_scores)

    print(f'Общее количество токенов в кластере: {summarize_tokens}')
    return clusters
