import codecs
import re
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Список моделей для embeddings
model_names=['intfloat/multilingual-e5-large',
             'sentence-transformers/all-MiniLM-L6-v2',
             'sentence-transformers/all-mpnet-base-v2',
             'cointegrated/rubert-tiny2']

# данные для теститрования: Вопрос, Фрагмент релевантного чанка (подстрока)
tests=[{"topic":"Студентам какая скидка?", "chunk":"Студентам скидка 15%"},
       {"topic":"алкоголь с собой?", "chunk":"При употребление алкогольных напитков действует ограничение"},
       {"topic":"курить у вас можно?", "chunk":"У нас не курят"},
       {"topic":"сколько стои аренда зала Гетсби?", "chunk":"4500 рублей в час вместимость до 36 человек включительно"},
       {"topic":"нужна ли сменная обувь?", "chunk":"принято переобуваться в сменную обувь — в свою или в наши тапочки"},
       ]

# Функция загрузки файла
def load_text(file_path):
    # Открытие файла для чтения
    with codecs.open(file_path, "r", encoding="utf-8", errors="ignore") as input_file:
        # Чтение содержимого файла
        content = input_file.read()
    return content

# Функция создания индексной базы знаний
def create_index_db(database, embeddings):
    source_chunks = []
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=0)
    for chunk in splitter.split_text(database):
        source_chunks.append(Document(page_content=chunk, metadata={}))

    chunk_num = len(source_chunks)
    print(f'chunk_num={chunk_num}')

    db = FAISS.from_documents(source_chunks, embeddings) # Создадим индексную базу из разделенных фрагментов текста
    return db

# Поиск релевантных отрезков из базы знаний
def get_message_content(topic, db):
    docs = db.similarity_search(topic, k = 2)
    message_content = re.sub(r'\n{2}', ' ', '\n '.join([f'\n#### Document excerpt №{i+1}####\n' + doc.page_content + '\n' for i, doc in enumerate(docs)]))
    return message_content

# Получение embeddings
def get_embeddings(model):
  # model_kwargs = {'device': 'cpu'}
  model_kwargs = {'device': 'cuda'}
  embeddings_hf = HuggingFaceEmbeddings(
    model_name=model,
    model_kwargs=model_kwargs
  )
  return embeddings_hf

# Функции для работы с файлом
def write_to_file(file_data, file_name):
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(file_data)

def append_to_file(new_line, file_name):
    with open(file_name, 'a', encoding='utf-8') as file:
        file.write('\n' + new_line)

# Основной цикл
def main():
    knowledge_base=load_text('Platon_KnowledgeBase_01.txt')

    res_file_name='embeddings_tests.csv'
    res_file_title = '"result";"model_name";"topic";"chunk"'
    write_to_file(res_file_title, res_file_name)

    rating_file_name='embeddings_rating.csv'
    ratings_file_title = '"model_rating";"model_name"'
    write_to_file(ratings_file_title, rating_file_name)

    for model in model_names:
      embeddings_hf = get_embeddings(model)
      idb= create_index_db(knowledge_base, embeddings_hf)
      model_rating = 0
      for t in tests:
        print(t)
        topic = t["topic"]
        chunk = t["chunk"]
        message_content = get_message_content(topic, idb)
        if chunk in message_content:
          res='Passed'
          model_rating +=1
        else:
          res='Failed'

        print(res, model, message_content)
        res_data_line = '"'+res+'";"'+model+'";"'+topic+'";"'+chunk+'"'
        # print(file_data_line)
        append_to_file(res_data_line, res_file_name)

      rating_data_line = '"'+str(model_rating)+'";"'+model+'"'
      append_to_file(rating_data_line, rating_file_name)

if __name__ == '__main__':
    main()