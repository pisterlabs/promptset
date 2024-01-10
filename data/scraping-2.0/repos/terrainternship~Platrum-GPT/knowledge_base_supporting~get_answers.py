import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import VectorStore, FAISS
import json
import os
import re

from llm_chat_bots import llm_init
from config import get_config

kb_store = get_config()['KnowledgeBase']['knowledge_base_storage']

prompt_system_path = get_config()['GenAnswers']['prompt_system']
llm_name = get_config()['GenAnswers']['llm_name']
api_key = get_config()['GenAnswers']['api_key']
model_name = get_config()['GenAnswers']['model_name']
tpe_max_workers = get_config()['GenAnswers']['tpe_max_workers']
answers_storage = get_config()['GenAnswers']['answers_storage']
similarity_search_k = get_config()['GenAnswers']['similarity_search_neighbours']

questions_storage = get_config()['GenQuestions']['questions_storage']


def process_answers():
    embedding_db = create_embedding_db()
    answers_from_llm(embedding_db)


def create_embedding_db():
    with open(kb_store + '/knowledge_base.json', 'r', encoding='utf-8') as file:
        knowledge_base = json.load(file)

    chunks = []
    for article in knowledge_base:
        for chunk in article['content']['chunks']:
            chunks.append(str(chunk['ru']))

    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = api_key

    return FAISS.from_texts(chunks, OpenAIEmbeddings())


def answers_from_llm(db: VectorStore):
    with open(questions_storage, 'r', encoding='utf-8') as file:
        questions = json.load(file)

    with open(prompt_system_path, 'r') as f:
        system = f.read().strip()

    data = {
        "frames": []
    }

    count = 0
    future_to_m = {}
    with ThreadPoolExecutor(max_workers=int(tpe_max_workers)) as executor:
        futures = []

        for chunk in questions['chunks']:
            chunk_id = chunk['id']
            article_id = chunk['article_id']

            for question in chunk['questions']:
                count += 1
                m = {
                    "article_id": article_id,
                    "chunk_id": chunk_id,
                    "question": question,
                    "answer": ""
                }
                docs = db.similarity_search(question, k=int(similarity_search_k))
                message_content = re.sub(r'\n{2}', ' ', '\n '.join(
                    [f'\nDocument excerpt№{i + 1}\n=====================' + doc.page_content + '\n' for i, doc in
                     enumerate(docs)]))

                config = {
                    'name': llm_name,
                    'api_key': api_key,
                    'model': model_name,
                    'system_message': system + "\n" + message_content
                }

                chat_bot = llm_init(config)

                future = executor.submit(lambda m=m: chat_bot.chat(question))
                futures.append(future)
                future_to_m[future] = m

        for future in as_completed(futures):
            m = future_to_m[future]

            try:
                result = future.result()
            except Exception as e:
                print(f"An error occurred during future execution: {e}")
                continue

            m['answer'] = result
            data['frames'].append(m)

            percent = round((count / questions['total_questions']) * 100)
            print(f"\rОбработка вопросов: выполнено {count} из {questions['total_questions']} ({percent}%)", end='')

    if data:
        with open(answers_storage, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
