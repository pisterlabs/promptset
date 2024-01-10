import os
import openai
import pandas as pd
import json
from dotenv import load_dotenv
import tiktoken

# Load environment variables from the file
load_dotenv()

#토큰 수 계산 함수
def encoding_getter(encoding_type: str):
    return tiktoken.encoding_for_model(encoding_type)

def tokenizer(string: str, encoding_type: str) -> list:
    encoding = encoding_getter(encoding_type)
    #print (encoding)
    tokens = encoding.encode(string)
    return tokens

def token_counter(string: str, encoding_type: str) -> int:
    num_tokens = len(tokenizer(string, encoding_type))
    return num_tokens


def read_concatenate_news(file_path, max_tokens=3000):
    news = pd.read_csv(file_path, delimiter='\t', header=None, names=['text'])
    concatenated_text = news['text'].str.cat(sep=' ')
    num_tokens = token_counter(concatenated_text, "gpt-3.5-turbo")
    print("토큰 수: " + str(num_tokens))

    if num_tokens >= max_tokens:
        tokens = tokenizer(concatenated_text, "gpt-3.5-turbo")
        concatenated_text = encoding_getter("gpt-3.5-turbo").decode(tokens[:max_tokens])
        return concatenated_text
    else:
        return concatenated_text

# def summarize_news(file_path):
#     query = read_concatenate_news(file_path)
#     GPT_MODEL = "text-davinci-003"  # 업데이트된 엔진 이름 사용

#     response = openai.Completion.create(
#         engine=GPT_MODEL,
#         prompt=f"뉴스에 대한 결과야. 요약설명해 {query}",
#         temperature=0.7,
#         max_tokens=150
#     )

#     response_message = response.choices[0].text.strip()
#     return response_message

def summarize_news(file_path):
    query = read_concatenate_news(file_path)
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"뉴스에 대한 결과야. 요약설명해 {query}"}]
    )
    return response.choices[0].message.content.strip()

# def summarize_news(file_path):
#     query = read_concatenate_news(file_path)
#     GPT_MODEL = "gpt-3.5-turbo"

#     messages = [
#         {"role": "system", "content": "You're the best summarizer. You have to show the right summary of the news. 모든 대답은 한글로."},
#         {"role": "user", "content": f"뉴스에 대한 결과야. 요약설명해 {query}"}
#     ]

#     response = openai.ChatCompletion.create(
#         model=GPT_MODEL,
#         messages=messages,
#         temperature=0
#     )

#     response_message = response.choices[0].message.content
#     return response_message

def save_to_json(result, srcText, node, output_file=None):
    if output_file is None:
        output_file = f'{srcText}_{node}_summary_result.json'

    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump({'summary': result}, json_file, ensure_ascii=False, indent=4)

    print(f"결과가 {output_file}에 저장되었습니다.")
    

# def main():
#     file_path = '고구마_naver_news_texts.txt'
#     result = summarize_news(file_path)
#     save_to_json(result)

# if __name__ == '__main__':
#     openai.api_key = os.environ.get("OPENAI_API_KEY")
#     main()
