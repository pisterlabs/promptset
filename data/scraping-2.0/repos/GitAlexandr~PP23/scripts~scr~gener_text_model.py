import requests
import pandas as pd
from docx import Document
from nltk import sent_tokenize, word_tokenize
import nltk
import openai
from openai.error import RateLimitError
import time

nltk.download('punkt')

openai.api_key = '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'

def process_messages(message):
    first_message = f"Задай краткий вопрос по следующему тексту:\n{message}"
    second_message = f"Переформулируй тремя различными способами следующий вопрос и напиши это списком:\n{first_message}"
    third_message = f"Сократи до 7 слов следующий текст:\n{message}"
    fourth_message = f"Переформулируй данный текст:\n{third_message}"

    questions = [first_message, second_message, third_message, fourth_message]

    list_response = []

    
    for question in questions:
            try:
                response = openai.Completion.create(
                    engine="text-davinci-002", 
                    prompt=question,
                    temperature=0.7,
                    max_tokens=150
                )
                result = response['choices'][0]['text']
                list_response.append(result)
                print(result)
            except RateLimitError as e:
                print(f"Rate limit exceeded. Waiting for 60 seconds before retrying...")
                time.sleep(60)
                # Retry the request
                response = openai.Completion.create(
                    engine="text-davinci-002", 
                    prompt=question,
                    temperature=0.7,
                    max_tokens=150
                )
                result = response['choices'][0]['text']
                list_response.append(result)
                print(result)

    main_theme = """ИНСТРУКЦИЯ ПО СИГНАЛИЗАЦИИ НА ЖЕЛЕЗНОДОРОЖНОМ ТРАНСПОРТЕ РОССИЙСКОЙ ФЕДЕРАЦИИ"""

    question = f"{list_response[0]}"
    questions_paraphrase = f"{list_response[1]}"    
    answer = f"{message}"
    answer_summary = f"{list_response[2]}"
    answer_paraphrase = f"{list_response[3]}"
    intent = "1"
    answers_merged = f"[{answer},{answer_summary},{answer_paraphrase}]"
    questions_merged = f"[{question},{questions_paraphrase}]"

    data = {
        'main_theme': [main_theme],
        'question': [question],
        'questions_paraphrase': [questions_paraphrase],
        'answer': [answer],
        'answer_summary': [answer_summary],
        'answer_paraphrase': [answer_paraphrase],
        'intent': [intent],
        'answers_merged': [answers_merged],
        'questions_merged': [questions_merged]
    }

    df = pd.DataFrame(data)

    xlsx_file = 'output.xlsx'
    df.to_excel(xlsx_file, index=False)

    print(f"Data written to {xlsx_file}")

def process_docx(docx_path):
    doc = Document(docx_path)
    doc_text = "\n".join([paragraph.text for paragraph in doc.paragraphs])

    sentences = sent_tokenize(doc_text)

    for i, sentence in enumerate(sentences, start=1):
        words_in_sentence = word_tokenize(sentence)
        if len(words_in_sentence) >= 8 and "Рисунок" not in words_in_sentence and "Методы" not in words_in_sentence and "Цвета" not in words_in_sentence and "Термины" not in words_in_sentence and "пункта" not in words_in_sentence:
            print(f"{i}. {sentence}")
            process_messages(sentence)


docx_path_to_process = 'data.docx'
process_docx(docx_path_to_process)
