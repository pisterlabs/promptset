import json
import time

import openai

from medspellchecker.tool_tests.metric_test.common.metric_test_with_context import MetricTestWithContext
from medspellchecker.tool_tests.metric_test.utils import EXTRA_SPACE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT, \
    MISSING_SPACE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT, SIMPLE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT

GPT4_RELATIVE_PATH_TO_SAVE_RESULT = "gpt4_fix.csv"
NUMBER_ELEMENTS_IN_BATCH = 10
TASK_DESCRIPTION_FOR_MODEL = """
Перепиши тексты, убрав орфографические ошибки, сохраняя исходный стиль текста, не меняя пунктуацию, не раскрывая аббревиатур, сохраняя заглавные и строчные буквы, не изменяя корректный текст, не добавляя знаки препинания. 
Напиши только правильный ответ без дополнительных объяснений. Напиши ответ в формате Json Array. 
Вот пункты:

"""

openai.api_key = None


def gpt4_tool_test(input_batches):
    batches = [input_batches[i:i + NUMBER_ELEMENTS_IN_BATCH]
               for i in range(0, len(input_batches), NUMBER_ELEMENTS_IN_BATCH)]

    raw_result = []
    timer = 0.0
    for idx, batch in enumerate(batches):
        print(f"Process batch #{idx}")
        prompt = TASK_DESCRIPTION_FOR_MODEL
        for anamnes in batch:
            prompt += ' '.join(anamnes) + "\n"

        start_time = time.time()
        chat_completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}])
        timer += time.time() - start_time

        json_result = json.loads(chat_completion.choices[0].message.content)
        for item in json_result:
            raw_result.append(item)

    return timer, raw_result


def perform_test():
    metric_test_with_context = MetricTestWithContext()
    return metric_test_with_context.compute_all_metrics(
        SIMPLE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT,
        MISSING_SPACE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT,
        EXTRA_SPACE_ERROR_TYPE_TO_DATA_PATH_WITH_CONTEXT,
        gpt4_tool_test)


if __name__ == '__main__':
    """
    Run test for GPT-4
    Link: https://platform.openai.com/docs/guides/gpt
    """
    test_result = perform_test()
    print(test_result)
