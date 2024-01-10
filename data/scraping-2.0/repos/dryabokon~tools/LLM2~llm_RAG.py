import warnings
warnings.filterwarnings("ignore")
import json
import uuid
import inspect
import os
# ----------------------------------------------------------------------------------------------------------------------
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
# ----------------------------------------------------------------------------------------------------------------------
import tools_time_profiler
import tools_Azure_Search
# ----------------------------------------------------------------------------------------------------------------------
class RAG(object):
    def __init__(self, chain,filename_config_vectorstore,vectorstore_index_name,filename_config_emb_model):
        self.TP = tools_time_profiler.Time_Profiler()
        self.chain = chain
        self.azure_search = tools_Azure_Search.Client_Search(filename_config_vectorstore,index_name=vectorstore_index_name,filename_config_emb_model=filename_config_emb_model)
        self.init_search_index(vectorstore_index_name, search_field='token')
        self.search_mode_hybrid = True
        self.init_cache()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def pdf_to_texts(self, filename_pdf):
        loader = PyPDFLoader(filename_pdf)
        pages = loader.load_and_split()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(pages)
        texts = [t for t in set([doc.page_content for doc in docs])]
        return texts
# ----------------------------------------------------------------------------------------------------------------------
    def file_to_texts(self,filename_in):
        with open(filename_in, 'r') as f:
            text_document = f.read()
        texts = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=500, chunk_overlap=100).create_documents([text_document])
        texts = [text.page_content for text in texts]
        return texts
# ----------------------------------------------------------------------------------------------------------------------
    def add_document_azure(self, filename_in, azure_search_index_name):
        print(filename_in)
        self.TP.tic(inspect.currentframe().f_code.co_name, reset=True)
        texts = self.pdf_to_texts(filename_in) if filename_in.split('/')[-1].split('.')[-1].find('pdf') >= 0 else self.file_to_texts(filename_in)
        docs =[{'uuid':uuid.uuid4().hex,'text':t} for t in texts]
        docs_e = self.azure_search.tokenize_documents(docs, field_source='text', field_embedding='token')

        if azure_search_index_name not in self.azure_search.get_indices():
            fields = self.azure_search.create_fields(docs_e, field_embedding='token')
            search_index = self.azure_search.create_search_index(azure_search_index_name, fields)
            self.azure_search.search_index_client.create_index(search_index)

        self.azure_search.search_client = self.azure_search.get_search_client(azure_search_index_name)
        self.azure_search.upload_documents(docs)
        self.TP.print_duration(inspect.currentframe().f_code.co_name)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def texts_to_docs(self,texts):
        docs = [Document(page_content=t, metadata={}) for t in texts]
        return docs
# ----------------------------------------------------------------------------------------------------------------------
    def do_docsearch_azure(self, query,azure_search_index_name,select='text',limit=4):
        self.azure_search.search_client = self.azure_search.get_search_client(azure_search_index_name)
        texts = self.azure_search.search_texts(query,select=select)
        docs = self.texts_to_docs(texts)
        return docs
# ----------------------------------------------------------------------------------------------------------------------
    def do_docsearch_azure_hybrid(self, query, azure_search_index_name, search_field='token', select='text',limit=4):
        self.azure_search.search_client = self.azure_search.get_search_client(azure_search_index_name)
        texts = self.azure_search.search_texts_hybrid(query,field=search_field,select=select,limit=limit)
        docs = self.texts_to_docs(texts)
        return docs
# ----------------------------------------------------------------------------------------------------------------------
    def pretify_string(self,text,N=120):
        lines = []
        line = ""
        for word in text.split():
            if len(line + word) + 1 <= N:
                if line:
                    line += " "
                line += word
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)

        result = '\n'.join(lines)

        return result
# ----------------------------------------------------------------------------------------------------------------------
    def init_search_index(self,azure_search_index_name,search_field):
        self.azure_search_index_name =azure_search_index_name
        self.search_field=search_field
        return
# ----------------------------------------------------------------------------------------------------------------------
    def init_cache(self):
        self.filename_cache = './data/output/' + 'cache.json'
        self.dct_cache = {}

        if os.path.isfile(self.filename_cache):
            with open(self.filename_cache, 'r') as f:
                self.dct_cache = json.load(f)
        return
# ----------------------------------------------------------------------------------------------------------------------
#     def Q(self,query,context_free=False,texts=None,q_post_proc=None,html_allowed=False):
#
#         if (texts is None or len(texts) ==0) and (query in self.dct_cache.keys()):
#             responce = self.dct_cache[query]
#         else:
#             if (texts is not None and len(texts) > 0):
#                 context_free = True
#                 #query+= '\nUse data below.\n'
#                 query+= '.'.join(texts)
#             if context_free:
#                 try:
#                     responce = self.chain.run({'question': query, 'input_documents': []})
#                 except:
#                     responce = ''
#             else:
#                 responce, texts= self.run_chain(query, azure_search_index_name=self.azure_search_index_name, search_field=self.search_field,select=self.text_key)
#
#             self.dct_cache[query] = responce
#             with open(self.filename_cache, "w") as f:
#                 f.write(json.dumps(self.dct_cache, indent=4))
#
#         if q_post_proc is not None:
#             responce = self.Q(f'{q_post_proc} Q:{query} A:{responce}', context_free=True)
#
#         if responce.find('Unfortunately') == 0 or responce.find('I\'m sorry') == 0 or responce.find('N/A') == 0:
#             #responce = "⚠️" + responce
#             responce = '&#9888 ' + responce
#
#         if html_allowed:
#             # responce = f'<span style="color: #086A6A;background-color:#EEF4F4">{query}</span>'
#             responce = f'<span style="color: #000000;background-color:#D5E5E5">{responce}</span>'
#
#         return responce
# ----------------------------------------------------------------------------------------------------------------------
    def run_query(self, query, search_field='token',select='text',limit=5):

        if self.search_mode_hybrid:
            docs = self.do_docsearch_azure_hybrid(query, self.azure_search_index_name,search_field=search_field,select=select,limit=limit)
        else:
            docs = self.do_docsearch_azure(query, self.azure_search_index_name,select=select)

        texts = [d.page_content for d in docs]

        try:
            response = self.chain.run(question=query,input_documents=docs)
        except:
            response = ''

        return response,texts
# ----------------------------------------------------------------------------------------------------------------------