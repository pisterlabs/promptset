from gpt_openai.gptfunc_a import *
from parsing.conf import config
import os 
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
# Инициализирум модель эмбеддингов
# self.price_fragments - сейчас всё будем хранить

from openai import AsyncOpenAI
import json

gpt = gptfunc_a_class()
# os.path.dirname(config.data_path)
class price_embeding_class:
    # таблица для хранения метаданных, изолируем
    ix_fragments =[]

    def __init__(self, data_path) -> None:
        self.data_path = data_path
        self.filename = data_path+"/price.md"
        self.filename_price2 = data_path+"/price2.md"
        self.db_filename = data_path+"/price_db"
        self.db_full_filename = data_path+"/full_db"
        self.db_filename2 = data_path+"/price_db2"
        self.data_key_file = data_path + "/data_key.json"
        self.data_full_key_file = data_path + "/full_key.json"
        self.db = None
        self.docs = []
        self.fragments = None
        self.price_fragments =None
        self.client = None
        self.embeddings = None

    def load_from_md_price_old(self):
        self.docs =[]
        with open(self.filename,"r", encoding="utf-8") as f:
            txt = f.read()
            self.docs.append(txt)
        self.fragments = gpt.split_text(self.docs)
        from langchain.docstore.document import Document
        def func(param:Document):
            h1 = param.metadata["H1"]
            h2 = param.metadata["H2"]
            # content =f.page_content
            d= Document(page_content=f"Стоимость. {h1}\n{h2}", metadata=param.metadata)
            return d
        self.price_fragments = [func(f)  for f in self.fragments]
        # Создадим индексную базу из разделенных фрагментов текста
        self.init_openai()
        self.db = FAISS.from_documents(self.price_fragments, self.embeddings)
        self.db.save_local(self.db_filename)

    def load_from_md_price(self):
        self.ix_fragments = []
        self.docs =[]
        self.price_fragments =[]
        self.load_from_md(self.filename, "Cтоимость досудебного исследования, Стоимость судебной экспертизы, руб. ", "57_1")
        self.load_from_md(self.filename_price2, "Стоимость независимой досудебной и судебной экспертизы Грузовые а/м и автобусы, Легковые а/м", "57_2")
    def save_keys(self):
        txt = json.dumps( self.ix_fragments, ensure_ascii=False)
        with open(self.data_key_file, "w", encoding="utf-8") as f:
            f.write(txt)

        fragments = [{"page_content":fr.page_content,"metadata":fr.metadata} for fr in self.price_fragments]
        txt = json.dumps( fragments, ensure_ascii=False)
        with open(self.data_path + "/fragments.json", "w", encoding="utf-8") as f:
            f.write(txt)

    def load_from_md(self, fn:str, add_text:str="",file_id:str="57"):
        '''Соберем ключи для поиска
        Заголовки первого и второго уровня
        Свяжем ключи поиска с контекстом таблиц
        '''
        h1items = {}
        new_fragments = []
        self.docs.clear()
        with open(fn,"r", encoding="utf-8") as f:
            txt = f.read()
            self.docs.append(txt)
        self.fragments = gpt.split_text(self.docs)
        
        key = len(self.price_fragments)
        p_key = key
        for f in self.fragments:
            h1 = f.metadata["H1"]
            ix = f.metadata["ix"]
            if h1 not in h1items:
                h1items[h1]=1
                fr ={"key":key, "ix":ix, "H1":h1, "type":"H1", "file_id":file_id}
                new_fragments.append(fr)
                d= Document(page_content=f"{add_text} {h1}",metadata={"key":key, "file_id":file_id})
                p_key = key
                self.price_fragments.append(d)
                key+=1
                if "H2" not in f.metadata:
                    fr["content"] = f.page_content
            if "H2" in f.metadata:
                h2 = f.metadata["H2"]
                fr ={"key":key, "pkey":p_key, "ix":ix, "type":"H2", "H1":h1, "H2":h2, "content":f.page_content, "file_id":file_id}
                new_fragments.append(fr)
                # content =f.page_content
                d= Document(page_content=f"{add_text} {h2}",metadata={"key":key, "H1":h1, "file_id":file_id})
                self.price_fragments.append(d)
                key+=1
        self.ix_fragments.extend(new_fragments)

    def faiss_from_md_save(self, db_filename):
        # Создадим индексную базу из разделенных фрагментов текста
        self.init_openai()
        self.db = FAISS.from_documents(self.price_fragments, self.embeddings)
        self.db.save_local(db_filename)


    def init_openai(self):
        self.client =AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        self.embeddings = OpenAIEmbeddings(async_client=self.client, openai_api_key=config.OPENAI_API_KEY)

    def load_db_price_old(self):
        if not os.path.exists(self.db_filename):
            self.load_from_md_price_old()
        else:
            self.init_openai()
            self.db = FAISS.load_local(self.db_filename, self.embeddings)

    def load_db_noprice(self):
        if not os.path.exists(self.db_filename2):
            self.load_from_md_price()
            self.faiss_from_md_save(self.db_filename2)
        else:
            self.load_db()

    def load_db(self):
        self.init_openai()
        self.db = FAISS.load_local(self.db_filename2, self.embeddings)
        with open(self.data_key_file, "r", encoding="utf-8") as f:
            self.ix_fragments = json.load(f)

    def search_with_score(self, search_str:str, limit_score:float = .8):
        docs = self.db.similarity_search_with_score(search_str, k=7)
        r = []
        score = 0
        def set_score(d, sc):
            d.metadata["score"]=sc
            return d
        for doc in docs:
            s = doc[1]
            if s==0:
                r.append( set_score(doc[0],s))
                break
            if score==0: 
                if s<.2:
                    score = s*1.7
                elif s<.3:
                    score = s +.05
                else:
                    score = s +.01
                if score>limit_score:
                    score = limit_score
            if s<score:
                r.append( set_score(doc[0],s))
        return r

    def parse_searched(self, docs):
        res = []
        h1keys =set()
        for doc in docs:
            key = doc.metadata["key"] #-1
            fr = self.ix_fragments[key]
            t = fr["type"]
            # if t=="H2":
            #     pkey = fr["pkey"]
            #     h1keys.add(pkey)
            if t=="H1":
                h1keys.add(key)
        di = {}
        for doc in docs:
            key = doc.metadata["key"] #-1
            fr = self.ix_fragments[key]
            t = fr["type"]
            if t=="H1":
                # Добавляем все дочерние
                if key in di:
                    ar=di[key]
                else:
                    ar=[]
                    di[key]=ar
                for fr_i in self.ix_fragments[key+1:]:
                    t2 = fr_i["type"]
                    if t2=="H1":
                        break
                    ar.append( Document(page_content=fr_i["H2"],metadata=fr_i))
            elif t=="H2":
                key = fr["pkey"]
                if not key in h1keys:
                    if key in di:
                        ar=di[key]
                    else:
                        ar=[]
                        di[key]=ar
                    ar.append( Document(page_content=fr["H2"],metadata=fr))
        for key in di:
            ar = di[key]
            for doc in ar:
                res.append(doc)
        return res

    def load_db_from_md(self):
        self.load_from_md_price()
        from parsing.SiteLoader import pages, dpages
        for i, e in enumerate(pages):
            if i==57:
                continue
            num = dpages[i]  # str(i).rjust(2,"0")
            num_i =str(i).rjust(2,"0")
            filename = num.md_file()
            print(filename)
            if os.path.exists(filename):
                self.load_from_md(filename,"", num_i)
        self.save_keys()

        self.faiss_from_md_save(self.db_filename2)

price_db = None

def create_price_db():
    global price_db
    price_db = price_embeding_class(config.data_path)
    price_db.load_db_noprice()
    return price_db


def create_db():
    global price_db
    price_db = price_embeding_class(config.data_path)
    price_db.load_db_from_md()
    return price_db

def load_db():
    global price_db
    price_db = price_embeding_class(config.data_path)
    price_db.load_db()
    return price_db

def to_zip():
    global price_db
    price_db = price_embeding_class(config.data_path)
    folder_to_zip = price_db.db_filename2
    output_filename = config.data_path+"/price_db2.zip"
    # Сохранение папки с векторной Базой знаний.
    # Архивирование папки с векторной Базой знаний
    import zipfile
    with zipfile.ZipFile(output_filename, 'w') as zip:
        for root, dirs, files in os.walk(folder_to_zip):
            for file in files:
                zip.write(os.path.join(root, file), arcname=file)
    print(f'База знаний - заархивирована. Имя файла - {output_filename}.')
