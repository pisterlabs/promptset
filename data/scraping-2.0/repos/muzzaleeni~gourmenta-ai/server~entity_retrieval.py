import openai
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def retrieve_entities_from_query(query):
    prompt = (
        f"Извлеките из запроса ниже следующие объекты:\n"
        f"Заведение, Кухня, Конкретная еда, Расстояние, Цена, Рейтинг.\n"
        f"Разделите каждую сущность разрывом строки. Поставьте знак минус, если вы не смогли найти информацию об обьекте. Не используйте никакой текст для обьектов с числовыми значениями!\n"
        f"Запрос для извлечения данных из: {query}\n"
        f"Отформатировать ответ как:\n"
        f"Заведение: ресторан\n Кухня: итальянская\n Конкретная еда: стейк\n Расстояние: 5\n Цена: 5000\n Рейтинг: 4.5\n"
        f"Переведите извлеченные обьекты на русский язык при надобности!"
        f"Даже если все обьекты не найдены, верните в таком же формате!"
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ],
        temperature=0,
        max_tokens=2000,
    )

    return response.choices[0].message.content
