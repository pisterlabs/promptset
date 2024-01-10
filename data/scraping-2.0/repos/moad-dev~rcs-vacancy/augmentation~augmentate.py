import openai
import csv
import json
import time
import os

openai.organization = os.environ.get('OPENAI_ORGANIZATION')
openai.api_key = os.environ.get('OPENAI_API_KEY')

# Define the prompt
system_prompt = (
    "Ты - аугментатор размеченных данных. Тебе будут даны размеченные данные в формате plaintext: тексты вакансий, в которых размечены:" 
    "1. Специализация (specialization ключ в json)"
    "2. Должностные обязанности (responsibilities ключ в json)"
    "3. Требования к соискателю (requirements)"
    "4. Условия (terms)"
    "5. Ключевые навыки (skills)"
    "6. Примечания (notes)"
    "Ты должен будешь провести аугментацию данных. Для каждой вакансии придумай новые две вакансии по следующим критериям:"
    "1. Текст оригинальной вакансии измененный так, чтобы он подходил по смыслу для другой придуманной тобой специальности в области строительства"
    "2. Текст оригинальной вакансии с сохранением специальности, но с перефразированными предложениями, заменами на синонимы, переформулировками."
    "Ты должен сохранить оригинальную разметку вакансий.  Выводи только новые размеченные вакансии в формате JSON, как массив объектов JSON [{}, {}, {}], без комментариев. Отсутствующие данные заполни по смыслу."
)

vacancies = []
with open('vacancies_original.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=';')
    for row in reader:
        vacancies.append(
            "Специализация:"
            + row.get('specialization', '-') + "\n"
            + "Должностные обязанности:"
            + row.get('responsibilities', '-') + "\n"
            + "Требования к соискателю:"
            + row.get('requirements', '-') + "\n"
            + "Условия:"
            + row.get('terms', '-') + "\n"
            + "Ключевые навыки:"
            + row.get('skills', '-') + "\n"
            + "Примечания:"
            + row.get('notes', '-') + "\n"
        )


for vacancy in vacancies:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": vacancy},
            ],
            temperature=1.05,
        )

        augmented = json.loads(response.choices[0].message.content)

        with open('vacancies_augmented.csv', 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=augmented[0].keys())

            for augmentation in augmented:
                print(augmentation)
                writer.writerow(augmentation)

        time.sleep(15)
    except Exception as e:
        print(e)
        time.sleep(15)
        continue
