#@title Вспомогательные функции
import tiktoken
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.prompts import PromptTemplate
import requests 
import re
import openai
from openai import OpenAI

class gpt_dbutil_class:
    def num_tokens_from_string(self, string: str, encoding_name: str) -> int:
        """Возвращает количество токенов в строке"""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def split_text(self, text, max_count, count_type, verbose=0):
        """ Функция разбивает текст на чанки. """

        # Функция для подсчета количества слов в фрагменте.
        def num_words(fragment):
            return  len(fragment.split())

        # Функция для подсчета количества токенов в фрагменте.
        def num_tokens(fragment):
            return self.num_tokens_from_string(fragment, "cl100k_base")

        # Разбивка на чанки происходит в два этапа.
        # На Первом этапе в формате Markdown делится на чанки по по делителям заголовков и подзаголовков.
        # На Втором  этапе полученные чанки делятся при необходимости на более мелкие.

        # Первый этап.
        # Шаблон для MarkdownHeaderTextSplitter по которому будет делится переданный текст в формате Markdown.
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        # Создаем экземпляр спилиттера.
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        # Получаем список чанков.
        fragments = markdown_splitter.split_text(text)

        # Второй этап.
        # Выбор функции подсчета длины в зависимости от типа подсчета.
        length_function = num_words if count_type == "words" else num_tokens
        # Создание объекта разделителя текста.
        splitter = RecursiveCharacterTextSplitter(chunk_size=max_count, chunk_overlap=0, length_function=length_function)
        # Список для хранения фрагментов текста.
        source_chunks = splitter.split_documents(fragments)

        # Обработка каждого фрагмента текста.
        if verbose:
            for chank in source_chunks:
                    # Вывод количества слов/токенов в фрагменте, если включен режим verbose.
                    count = length_function(chank.page_content)
                    answer = f"{count_type} in text fragment = {count}\n{'-' * 5}\n{chank}\n{'=' * 20}"
                    print(self.insert_newlines(answer))

        # Возвращение списка фрагментов текста.
        return source_chunks


    def create_embedding(self, data, max_count, count_type, verbose=0):
        source_chunks = []
        source_chunks = self.split_text(text=data, max_count=max_count, count_type=count_type, verbose=verbose)

        # Создание индексов документа
        search_index = FAISS.from_documents(source_chunks, OpenAIEmbeddings(), )

        count_token = self.num_tokens_from_string(' '.join([x.page_content for x in source_chunks]), "cl100k_base")
        print('\n ===========================================: ')
        print('Количество токенов в документе :', count_token)
        print('ЦЕНА запроса:', 0.0004*(count_token/1000), ' $')
        return search_index

    def load_search_indexes(self, url: str, max_count, count_type, verbose=0) -> str:
        """Download the document as plain text."""
        try:
            response = requests.get(url) # Получение документа по url.
            response.raise_for_status()  # Проверка ответа и если была ошибка - формирование исключения.
            # Создание векторной База знаний.
            search_index = self.create_embedding(
                            response.text,
                            max_count=max_count,
                            count_type=count_type,
                            verbose=verbose)
            return search_index
        except Exception as e:
            print(e)

    def load_file(self, url: str) -> str:
        try:
            response = requests.get(url) # Получение документа по url.
            response.raise_for_status()  # Проверка ответа и если была ошибка - формирование исключения.
            return response.text
        except Exception as e:
            print(e)

    def num_tokens_from_messages(self, messages, model="gpt-3.5-turbo-0301"):
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
            num_tokens = 0
            for message in messages:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens += -1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens
        else:
            raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.""")

    def insert_newlines(self, text: str, max_len: int = 120) -> str:
        """ Функция форматирует переданный текст по длине
        для лучшего восприятия на экране."""
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line + " " + word) > max_len:
                lines.append(current_line)
                current_line = ""
            current_line += " " + word
        lines.append(current_line)
        return "\n".join(lines)

    def answer_index(self, system, topic, search_index, temp = 1, verbose = 0, top_similar_documents = 5):
        """ Основная функция которая формирует запрос и получает ответ от OpenAI по заданному вопросу
        на основе векторной Базы знаний. """

        # Выборка чанков по схожести с вопросом.
        docs = search_index.similarity_search(topic, k=top_similar_documents)

        # Формирование контекста на основе выбранных чанков.
        message_content = re.sub(r'\n{2}', ' ', '\n '.join([f'\nОтрывок документа №{i+1}\n=====================' + doc.page_content + '\n' for i, doc in enumerate(docs)]))

        if (verbose):
            print('\n ===========================================: ')
            print('message_content :\n ======================================== \n', message_content)
        # Формирование запроса к OpenAI.
        messages = [
            {"role": "system", "content": system + f"{message_content}"},
            {"role": "user", "content": topic}
        ]
        # Отправка запроса к Open AI.
        client = OpenAI(
            api_key=openai.api_key
        )
        completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=temp
        )
        # Вывод на экран подробностей при необходимости.
        if (verbose):
            print('\n ===========================================: ')
            print(f'{completion.usage.prompt_tokens} токенов использовано на вопрос.')
            print('\n ===========================================: ')
            print(f'{completion.usage.total_tokens} токенов использовано всего (вопрос-ответ).')
            print('\n ===========================================: ')
            print('ЦЕНА запроса с ответом :', 0.002*(completion.usage.total_tokens/1000), ' $')
            print('\n ===========================================: ')
        # Ответ OpenAI.
        answer = 'ОТВЕТ : \n' + self.insert_newlines(completion.choices[0].message.content)
        return answer

