from openai import OpenAI, OpenAIError
import pandas as pd

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="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",
)

csv_file_path = 'Data/Input.csv'

# Открываем CSV файл
input_table = pd.read_csv(csv_file_path, sep=',', encoding='UTF32')

# Проходим циклом по каждой строке в файле
for row in input_table.iterrows():
        try:
            # Проверяем, что вторая и третья колонки пусты
            if pd.isnull(row[1]["result"]):
                user_input = (
                        "Представь, что ты специалист в области data science. Ты получаешь сырой текст"
                        +str(row[1]["text"])+
                        "и тебе необходимо на основе информации придумать вопросы, на который содержимое страницы может дать краткий, чётко структурированный ответ. Пришли мне выход в формате ""Вопрос:.... Ответ:..."
                       #"Привет, как дела, ты работаешь?"
                        )
                # Вызов API ChatGPT для выполнения запроса
                response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": user_input,
                        }
                    ],
                    model="gpt-3.5-turbo"
                )
                row[1]["result"] = response.choices[0].message.content
                # Сохранение новой таблицы в CSV-файл
                input_table.to_csv(csv_file_path, index=False, encoding='UTF32')
        except OpenAIError as e:
            # Обработка ошибок OpenAI API
            row[1]["result"] = f"Ошибка API: {str(e)}"
            input_table.to_csv(csv_file_path, index=False, encoding='UTF32')
        except Exception as e:
            # Общая обработка других исключений
            row[1]["result"] = f"Ошибка: {str(e)}"
            input_table.to_csv(csv_file_path, index=False, encoding='UTF32')

