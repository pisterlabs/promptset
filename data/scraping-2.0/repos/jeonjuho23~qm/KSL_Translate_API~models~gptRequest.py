import openai
from config.configkey import API_KEY_OPENAI## 주호꺼
from config.configkey import API_KEY_OPENAI_JAC ## 내꺼

openai.api_key = API_KEY_OPENAI_JAC


def extract_meaningful_words(words):
    text = """당신의 임무는 다음 작업을 순서대로 수행하는 것입니다.
    1 - 백틱 세 개로 구분된 다음 텍스트의 맞춤법을 검사하여 수정합니다.
    2 - 수정된 텍스트를 유의미한 관용어구 혹은 단어로 나누어 문자열의 배열 형태로 출력합니다.

    다음 형식을 사용합니다:
    수정된 텍스트: <맞춤법 수정>
    출력: <수정된 텍스트를 관용어구, 단어, 물음표로 나눈 문자열>

    텍스트: ```""" + words + """```
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": text}
        ],
        max_tokens=100,
        temperature=0.1,
    )

    answer = response.choices[0].message.get('content', '')
    print(answer)
    lines = answer.split("\n")
    output_array = lines[1].split(":")[1]
    output_array = [word.strip() for word in output_array.split(",")]

    final_output_array = []
    for word in output_array:
        if '[' in word or ']' in word:
            inner_array = word.strip('[').strip(']').strip().split(' ')
            final_output_array.extend(inner_array)
        else:
            final_output_array.append(word)
    print(final_output_array)

    return final_output_array
