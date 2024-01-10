from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import asyncio
import os
from collections import Counter
import re
from datetime import datetime
import pytz


def extract_top_entities(input_documents, given_query, custom_file_path='custom_entities.txt'):
    # カスタム辞書をテキストファイルから読み込む
    try:
        with open(custom_file_path, 'r') as f:
            custom_dictionary = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        custom_dictionary = []
    # 空のリストを用意
    bigrams = []
    # 各ドキュメントに対してbigramを抽出
    for doc in input_documents:
        bigrams.extend(re.findall(r'\b\w+\s+\w+\b', doc.page_content))
    # かぎ括弧で囲まれている固有表現を抽出
    bracketed_entities = []
    for doc in input_documents:
        bracketed_entities.extend(re.findall(r'「(.*?)」', doc.page_content))
    # 頻度が多い固有表現をカスタム辞書に保存
    entity_freq = Counter(bracketed_entities)
    new_entities = [entity for entity, freq in entity_freq.items() if freq > 0]
    # カスタム辞書と結合
    top_terms = list(set(bracketed_entities) | set(new_entities) | set(bigrams) | set(custom_dictionary))
    # クエリに固有表現を追加
    modified_query = given_query
    entities = []
    for term in top_terms:
        if term in modified_query:
            modified_query = modified_query.replace(term, f"「{term}」")
            entities.append(term)
    return modified_query, entities


class RetrievalQAFromFaiss:
    def __init__(self):
        self.message_histories = {}
        self.total_tokens = 0
        self.input_txt = ""

    async def GetAnswerFromFaiss(self, initial_query, user_key):
        self.input_txt = initial_query
        embeddings = OpenAIEmbeddings()
        # 現在の日付と時刻を取得します（日本時間）。
        now = datetime.now(pytz.timezone('Asia/Tokyo'))
        # 年、月、日を取得します。
        year = now.year
        month = now.month
        day = now.day
        # 直近のメッセージを取得
        try:
            recent_messages = self.message_histories[user_key][-4:]  # 2往復分なので、最後の4メッセージ
        except KeyError:
            recent_messages = []
        # 対話形式に変換
        dialogue_format = ""
        for msg in recent_messages:
            role = "User" if msg['role'] == 'user' else "Assistant"
            dialogue_format += f"{role}: {msg['content']}\n"
        print("dialogue_format: ", dialogue_format)
        if os.path.exists("./faiss_index"):
            docsearch = FAISS.load_local("./faiss_index", embeddings)
            refine_prompt_template = (
                    f"Today is the year {year}, the month is {month} and the date {day}."
                    f"The current time is {now}." + "\n"
                    "The original question is as follows:\n {question}\n"
                    "We have provided an existing answer:\n {existing_answer}\n"
                    "Please refine the above answer using the context information below "
                    "(Only if needed and relevant to the question).\n"
                    "------------\n"
                    "{context_str}\n"
                    "------------\n"
                    "Use the provided context to refine your responses if it makes them more concise, "
                    "direct, and relevant to the original question. "
                    "If not, return the existing answer without any changes."
            )
            refine_prompt = PromptTemplate(
                input_variables=["question", "existing_answer", "context_str"],
                template=refine_prompt_template,
            )
            initial_qa_template = (
                "Here is the context: \n"
                "---------------------\n"
                "{context_str}"
                "\n---------------------\n"
                "Please answer the following question based on the context: {question}. "
                "If you can't find the answer, Please respond that you don't know to within your knowledge."
                "Your answer should be in Japanese."
            )
            initial_qa_prompt = PromptTemplate(
                input_variables=["context_str", "question"], template=initial_qa_template
            )
            qa_chain = load_qa_chain(
                ChatOpenAI(temperature=0, model_name="gpt-4-0613", max_tokens=500),
                chain_type="refine",
                question_prompt=initial_qa_prompt,
                refine_prompt=refine_prompt
            )
            similar_documents = docsearch.similarity_search(query=initial_query)
            modified_ver_query, entities = extract_top_entities(similar_documents, initial_query)
            print("modified_ver_query: ", modified_ver_query)
            print("entities: ", entities)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: qa_chain(
                {"input_documents":
                 similar_documents,
                 "question": modified_ver_query},
                return_only_outputs=True))
            # responseオブジェクトからanswerとsource_urlを抽出
            try:
                answer = response['output_text']
            except (TypeError, KeyError, IndexError):
                answer = "APIからのレスポンスに問題があります。開発者にお問い合わせください。"
            print("answer: ", answer)
            return answer, self.input_txt
        else:
            answer = "申し訳ありません。データベースに不具合が生じているようです。開発者にお問い合わせください。"
            return answer, self.input_txt
