# test
import openai
import time

# 발급받은 API 키 설정
OPENAI_API_KEY = "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"

# openai API 키 인증
openai.api_key = OPENAI_API_KEY

def word_dff_jury(word_a, word_b, model="gpt-4", max_try=3):
    # 모델 - GPT 3.5 Turbo 선택
    # model = "gpt-3.5-turbo"

    # 질문 작성하기
    # instruction = "You are a machine that answers two words {word A, word B} by choosing the one you expect to learn at a younger age. When answering, please consider all of the following conditions: 1. Answer by choosing the word that is expected to be learned at a younger age 2. Please indicate your level of confidence in your answer as a percentage (100%=very confident, 0%=not confident at all) example = word A, 70% 3. Answer concisely without further explanation."
    # instruction = "You are tasked with comparing two words: {word A, word B}. Choose the word that is typically learned at a younger age. When answering, follow these guidelines: 1. Select the word expected to be learned earlier in life. 2. State your confidence in your choice as a percentage (e.g., 70% means you are 70% confident in your choice). 3. Respond with only the chosen word and your confidence percentage without adding any other explanations. Example response: Word A, 70%"
    instruction = "You are to choose between two words: {word A, word B}. Pick the word you believe is learned first in childhood. When answering: 1.Choose the earlier-learned word. 2.Give your confidence level as a percentage (e.g., 'word A, 70%'). 3.Only state the word and percentage, nothing more."
    query = "{" + word_a + "," + word_b + "}"

    # 메시지 설정하기
    messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": query}
    ]

    # print("model name:", model)

    # 비용 계산
    # cost_sum = float()

    # ChatGPT API 호출하기
    while 1:

        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0,
                request_timeout=15,
            )

            answer = response['choices'][0]['message']['content']

            # 비용
            # input_token_num = response["usage"]["prompt_tokens"]
            # output_token_num = response["usage"]["completion_tokens"]
            # cost = float(input_token_num) / 1000 * 0.0015 + float(output_token_num) / 1000 * 0.002
            # cost_won = cost * 1330 # $ to won
            #
            return answer

        # except Exception as e:
        #     if e:
        #         # print(e)
        #         # print('Timeout error, retrying...')
        #         max_try -= 1
        #         time.sleep(1)
        #     else:
        #         raise e

