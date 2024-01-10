import os
import openai
import pandas as pd
import json
from dotenv import load_dotenv

# Load environment variables from the file
load_dotenv()

def read_concatenate_news(file_path):
    news = pd.read_csv(file_path, delimiter='\t', header=None, names=['text'])
    concatenated_text = news['text'].str.cat(sep=' ')
    return concatenated_text

def summarize_news(file_path):
    query = read_concatenate_news(file_path)
    GPT_MODEL = "gpt-3.5-turbo"

    messages = [
        {"role": "system", "content": "You're the best summarizer. You have to show the right summary of the news. 모든 대답은 한글로."},
        {"role": "user", "content": f"뉴스에 대한 결과야. 요약설명해 {query}"}
    ]

    response = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages=messages,
        temperature=0
    )

    response_message = response.choices[0].message.content
    return response_message

def save_to_json(result, srcText, node, output_file=None):
    if output_file is None:
        output_file = f'{srcText}_{node}_summary_result.json'

    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump({'summary': result}, json_file, ensure_ascii=False, indent=4)

    print(f"결과가 {output_file}에 저장되었습니다.")


# def save_to_json(result, output_file='summary_result.json'):
#     with open(output_file, 'w', encoding='utf-8') as json_file:
#         json.dump({'summary': result}, json_file, ensure_ascii=False, indent=4)
#     print(f"결과가 {output_file}에 저장되었습니다.")

def main():
    file_path = '오늘날씨_naver_news_texts.txt'
    result = summarize_news(file_path)
    save_to_json(result)

if __name__ == '__main__':
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    main()
