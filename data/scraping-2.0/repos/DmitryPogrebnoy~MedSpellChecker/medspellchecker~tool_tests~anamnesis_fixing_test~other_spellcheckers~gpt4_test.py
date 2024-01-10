import json
import os

import openai
import pandas as pd

GPT4_RELATIVE_PATH_TO_SAVE_RESULT = "gpt4_fix.csv"
NUMBER_ELEMENTS_IN_BATCH = 10
TASK_DESCRIPTION_FOR_MODEL = """
Перепиши тексты, убрав орфографические ошибки, сохраняя исходный стиль текста, не меняя пунктуацию, не раскрывая аббревиатур, сохраняя заглавные и строчные буквы, не изменяя корректный текст, не добавляя знаки препинания. 
Напиши только правильный ответ без дополнительных объяснений. Напиши ответ в формате Json Array. 
Вот пункты:

"""

openai.api_key = None


def perform_anamnesis_fixing_test(test_function, relative_path_to_save_result):
    path_to_test_dataset = os.path.join(os.path.dirname(__file__),
                                        "../../../../data/test/anamnesis_fixing_test/test_ru_med_prime_data.csv")
    df = pd.read_csv(path_to_test_dataset)
    anamnesis = df["data"].values[:100]
    result = test_function(anamnesis)
    answer_df = pd.DataFrame(result, columns=["data"])
    absolute_path_to_save_result = os.path.join(os.path.dirname(__file__),
                                                "../../../../data/test/anamnesis_fixing_test/after_fix/",
                                                relative_path_to_save_result)
    answer_df.to_csv(absolute_path_to_save_result)
    print(f"Result saved to {absolute_path_to_save_result}")


def gpt4_tool_test(input_batches):
    batches = [input_batches[i:i + NUMBER_ELEMENTS_IN_BATCH]
               for i in range(0, len(input_batches), NUMBER_ELEMENTS_IN_BATCH)]

    raw_result = []
    for idx, batch in enumerate(batches):
        print(f"Process batch #{idx}")
        prompt = TASK_DESCRIPTION_FOR_MODEL
        for anamnes in batch:
            prompt += anamnes + "\n"

        chat_completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}])

        raw_result.append(json.loads(chat_completion.choices[0].message.content))

    return raw_result


if __name__ == '__main__':
    """
    Run test for GPT-4
    Link: https://platform.openai.com/docs/guides/gpt
    """
    perform_anamnesis_fixing_test(gpt4_tool_test, GPT4_RELATIVE_PATH_TO_SAVE_RESULT)
