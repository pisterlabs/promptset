"""
Модуль перевода текстовых файлов в индексы
"""

import os

import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from create_bot import OPENAI_API_KEY, FAISS_DB_DIR
from config import TXT_DB_DIR, ROOT_DIR
import tiktoken


os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY
txt_db_dir = os.path.join(ROOT_DIR, TXT_DB_DIR)
faiss_db_dir = os.path.join(ROOT_DIR, FAISS_DB_DIR)  # полный путь до индексов


def get_file_name(file_path):
    """Получение имени файла без расширения
    """
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    return file_name


def get_files(path=txt_db_dir):
    """
    Получение списка файлов в указанной директории для подготовки индексов
    :param path:
    :return:
    """
    files = []
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            files.append(os.path.join(path, file))
    return files


def num_tokens_from_string(string: str):
      encoding = tiktoken.get_encoding("cl100k_base")
      num_tokens = len(encoding.encode(string))
      return num_tokens


def create_faiss_indexes(path=txt_db_dir):
    all_txt_files = get_files()
    print(f'create_faiss_indexes: Проверим файлы в директории "{TXT_DB_DIR}": \n {all_txt_files}')

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["##_", "\n\n", "\n", " ", ""],
        chunk_size=1024,
        length_function=lambda x: num_tokens_from_string(x))  # создали Сплиттер для разбиения на чанки
    
    embeddings = OpenAIEmbeddings()  # по умолчанию использует самую дешевую модель 'text-embedding-ada-002'

    for file in all_txt_files:
        print(f'Имя файла {get_file_name(file)}')
        local_path_to_base = os.path.join(faiss_db_dir, get_file_name(file))
        loader = TextLoader(file, encoding='utf-8')  # загрузили текстовый файл в loader
        document = loader.load()  # сделали из него документ
        split_doc = text_splitter.split_documents(document)  # с помощью сплиттера разбили документ на чанки
        print(f'Количество чанков: {len(split_doc)}')
        db = FAISS.from_documents(split_doc, embeddings)
        db.save_local(local_path_to_base)


def get_bases():
    faiss_db_dir = os.path.join(ROOT_DIR, FAISS_DB_DIR)
    list_indexes = []
    for folder in os.listdir(faiss_db_dir):
        if os.path.isdir(os.path.join(faiss_db_dir, folder)):
            list_indexes.append(os.path.basename(folder))
    print(f'get_bases: Список баз индексов: {list_indexes}')
    return list_indexes


def read_faiss_indexes(indexes):
    """
    Чтение индексов из всех индексных файлов
    :param path: локальный путь в проекте до папки с индексами
    :return: база индексов
    """
    db_path = os.path.join(ROOT_DIR, FAISS_DB_DIR)
    flag = True     # Признак первой базы для чтения. Остальные базы будем добавлять к имеющейся
    # Перебор всех курсов в списке courses:
    print(f'Старт read_faiss_indexes: {indexes =}')
    count_base = 0  # сосчитаем количество курсов
    for index_file in indexes:
        index_path = os.path.join(db_path, index_file)  # получаем полный путь к курсу
        print(f'read_faiss_indexes - ищем индекс {count_base}: {index_file =}, {index_path =}')
        count_base += 1
        if flag:
            # Если flag равен True, то создается база данных FAISS из папки index_path
            db = FAISS.load_local(index_path, OpenAIEmbeddings())
            flag = False
            print(f'read_faiss_indexes: прочитали новый индекс')
        else:
            # Иначе происходит объединение баз данных FAISS
            db.merge_from(FAISS.load_local(index_path, OpenAIEmbeddings()))
            print(f'read_faiss_indexes: Добавили в индекс')

    return db

if __name__ == '__main__':
    # Проверка созданного индекса
    ''' 
    загрузим сохраненный индекс в новую базу индексов и проведем по ней поиск
    '''

    create_faiss_indexes()
    faiss_indexes = get_bases()
    db = read_faiss_indexes(faiss_indexes)

    query1 = "Какие самые главные принципы в управлении проектами?"
    docs1 = db.similarity_search(query1)
    print(f"{query1 =}")
    print(docs1[0].page_content)

    # query2 = "Импортный крепеж"
    # docs2 = db.similarity_search(query2)
    # print(f"{query2 =}")
    # print(docs2[0].page_content)
"""
    # Uncomment the following line if you need to initialize FAISS with no AVX2 optimization
    # os.environ['FAISS_NO_AVX2'] = '1'

    # база знаний хранится в файле:
    txt_base = TXT_DB_DIR

    txt_base = '/home/alex/Documents/agent5/k_base/standart_mop.txt'
    print(f'Имя файла {get_file_name(txt_base)}')

    # Создадим путь для хранения индексов (создадим каталог с именем файла)
    faiss_db_dir = FAISS_DB_DIR
    local_path_to_base = os.path.join(faiss_db_dir, get_file_name(txt_base))

    loader = TextLoader(txt_base)   # загрузили текстовый файл в loader
    documents = loader.load()       # сделали из него документ
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)     # создали Спилттер для разбиения на чанки

        # TODO: RecursiveCharacterTextSplitter - разделение текста на части с помощью одного типа символов (по умолчанию - “nn”), но с возможностью рекурсивного разделения
        # TODO: CodeMarkupTextSplitter - разделение текста на части с помощью маркеров кода и разметки

    docs = text_splitter.split_documents(documents)     # с помощью сплиттера разбили документ на чанки

    # создание и сохранение базы индексов
    embeddings = OpenAIEmbeddings()     # по умолчанию использует самую дешевую модель 'text-embedding-ada-002'
    print(f'{os.getcwd() = }\n{local_path_to_base =}')
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(local_path_to_base)


    new_db = FAISS.load_local(local_path_to_base, embeddings)
    query = "расскажи про менеждера по продажам"
    docs = new_db.similarity_search(query)

    print(f"{query =}")
    print(docs[0].page_content) 
"""