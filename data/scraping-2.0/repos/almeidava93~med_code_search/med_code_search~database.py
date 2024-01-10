import pandas as pd
from google.cloud import firestore
import streamlit as st
from rank_bm25 import BM25Okapi
import spacy
from unidecode import unidecode
import streamlit as st
import uuid
from datetime import datetime as dt
import openai
from openai.embeddings_utils import get_embedding
import pinecone
from functools import lru_cache


#IMPORTANT VARIABLES TO BE USED
service_account_info = st.secrets["gcp_service_account_firestore"]
firebase_storage_config = st.secrets["gcp_service_account"]
api_key = st.secrets['openai']['key']
openai.api_key = api_key

# Load Pinecone API key
api_key = st.secrets['pinecone']['key']
pinecone.init(api_key=api_key, environment='us-east1-gcp')
index = pinecone.Index("icpc-embeddings")

@st.cache_resource
def load_firestore_client(service_account_info = service_account_info):
  firestore_client = firestore.Client.from_service_account_info(service_account_info)
  return firestore_client

firestore_client = load_firestore_client() #Carrega a conexão com a base de dados com cache.



@st.cache_data
def firestore_query(firestore_client = firestore_client, field_paths = [], collection = 'tesauro'):
  #Load dataframe for code search
  firestore_collection = firestore_client.collection(collection)
  filtered_collection = firestore_collection.select(field_paths)#Fields containing useful data for search engine
  filtered_collection = filtered_collection.get() #Returns a list of document snapshots, from which data can be retrieved
  filtered_collection_dict = [doc.to_dict() for doc in filtered_collection] #Returns list of dictionaries 
  filtered_collection_dataframe = pd.DataFrame.from_records(filtered_collection_dict) #Returns dataframe
  return filtered_collection_dataframe



#OTHER VARIABLES OF INTEREST
search_code_data = firestore_query(field_paths=['text', '`text with special characters`'])
search_code_data_cid = firestore_query(field_paths=['CAT', 'DESCRICAO', '`Termo Português`'], collection='tesauro_cid')
ciap_criteria = firestore_query(field_paths=['code','`inclusion criteria`', '`exclusion criteria`'], collection='ciap_criteria')

ciap_df = firestore_query(field_paths=['`CIAP2_Código1`', '`titulo original`']).agg(" | ".join, axis=1).drop_duplicates()
ciap_list = list(ciap_df)



#Função que gera o índice BM25 para a busca e atualiza o arquivo
@st.cache_resource
def bm25_index(data = search_code_data['text'].astype(str)):
    #Launch the language object
    nlp = spacy.blank("pt")
    #Preparing for tokenisation
    text_list = data.str.lower().values
    tok_text=[] # for our tokenised corpus
    #Tokenising using SpaCy:
    for doc in nlp.pipe(text_list, disable=["tagger", "parser","ner"]):
        tok = [t.text for t in doc]
        tok_text.append(tok)
    #Building a BM25 index
    bm25 = BM25Okapi(tok_text)
    return bm25


#OTHER VARIABLES
bm25 = bm25_index()


#Função que retorna o código escolhido
def search_code(input, n_results, data = search_code_data, bm25=bm25):
    if input != "":
        #Generate search index
        #bm25 = bm25_index()
        #Querying this index just requires a search input which has also been tokenized:
        input = unidecode(input) #remove acentos e caracteres especiais
        tokenized_query = input.lower().split(" ")
        results = bm25.get_top_n(tokenized_query, data.text.values, n=n_results)
        results = [i for i in results]
        selected_code = st.radio('Esses são os códigos que encontramos. Selecione um para prosseguir.', results, index=0, help='Selecione um dos códigos para prosseguir.')
        return selected_code

@st.cache_data
def join_columns(dataframe, column_names, delimiter=' | ', drop_duplicates=False):
  df = dataframe[column_names].agg(delimiter.join, axis=1)
  if drop_duplicates==True: df.drop_duplicates()
  return df


#Função que remove caracteres especiais de uma coluna de um dataframe
@st.cache_data
def unidecode_df(dataframe, column_names):
  return dataframe[column_names].apply(lambda x: unidecode(x))


#Preparing data for CID10 search
##Create dataframe and a merged column for a multiselect option list
search_code_data_cid_multiselect = pd.DataFrame()
search_code_data_cid_multiselect['text'] = join_columns(search_code_data_cid, column_names=['CAT', 'DESCRICAO', 'Termo Português'])
##Create a column with the same data, but unidecoded, for bm25 index creation
search_code_data_cid['text'] = unidecode_df(search_code_data_cid_multiselect, column_names='text')
##Create bm25 index for CID10 search
bm25_cid = bm25_index(data = search_code_data_cid['text'])


#Função que salva os dados na base de dados
#@st.cache(hash_funcs={firestore.Client: id}, ttl=None, show_spinner=True)
def save_search(text_input, n_records, n_results, selected_code, collection_name, firestore_client=firestore_client):
  #input -> text input for code search
  #n_records -> number of records searched
  #n_results -> number of results shown
  #selected_code -> selected code in radio button
  
  assert type(text_input) == str

  search_id = 'search_id_' + str(uuid.uuid4()) #id for document name
  datetime = dt.now() #date and time of search
  
  ##Saving data:
  doc_ref = firestore_client.collection(collection_name).document(search_id)
  doc_ref.set({
            'search id': search_id,
            'text input': text_input,
            'timestamp': datetime,
            'n records searched': n_records,
            'n results shown': n_results,
            'selected code': selected_code
        })
        
#Função que salva inputs quando eles mudam
def save_on_change(key: str, collection: str):
    doc_ref = firestore_client.collection(collection)
    input_id = 'input_id_' + str(uuid.uuid4()) #id for document name
    datetime = dt.now() #date and time of search
    doc_ref.document(input_id).set(
            {
                key: st.session_state[key],
                "timestamp": datetime
            },
            merge=True
        )

#Função que resgata da base dados critérios de inclusão e exclusão
#@st.cache(ttl=None, show_spinner=True)
def get_code_criteria(code: str) -> dict[str, str]:
  code_criteria = ciap_criteria[ciap_criteria['code']==code].iloc[0].to_dict()
  return code_criteria



@st.cache_data
def get_code_criteria_from_firebase(code) -> pd.DataFrame:
    if type(code)==str:
        code = [code]

    if type(code)==pd.core.series.Series:
        code = list(code)

    if type(code)==pd.DataFrame:
        try:
            code = list(code['code'])
        except KeyError:
            code = list(code['CIAP2_Código1'])

    if type(code)==list:
        query = firestore_client.collection('ciap_criteria').where('code','in', code).get()
        docs_list = [doc.to_dict() for doc in query]
        docs_df = pd.DataFrame.from_records(docs_list)
        docs_df.index = docs_df['code']
        docs_df.drop('code', axis=1, inplace=True)
        return docs_df
    else:
        raise TypeError("This function expects a string with one ICPC-2 code or a iterable (list, pandas series, dataframe with 'code' column) of strings, each one corresponding to an ICPC-2 code")


@st.cache_data
def get_code_data_from_firebase(code, fields=['codigo','`titulo original`','`considerar`','`critérios de inclusão`', '`critérios de exclusão`']) -> pd.DataFrame:
    if type(code)==str:
        code = [code]

    if type(code)==pd.core.series.Series:
        code = list(code)

    if type(code)==pd.DataFrame:
        try:
            code = list(code['code'])
        except KeyError:
            code = list(code['CIAP2_Código1'])

    if type(code)==list:
        query = firestore_client.collection('ciap_data').select(field_paths=fields).where('codigo','in', code).get()
        docs_list = [doc.to_dict() for doc in query]
        docs_df = pd.DataFrame.from_records(docs_list)
        docs_df.index = docs_df['codigo']
        docs_df.drop('codigo', axis=1, inplace=True)
        return docs_df
    else:
        raise TypeError("This function expects a string with one ICPC-2 code or a iterable (list, pandas series, dataframe with 'code' column) of strings, each one corresponding to an ICPC-2 code")


@st.cache_data
def get_input_embedding(input):
    input_vector = get_embedding(input, engine="text-embedding-ada-002")
    return input_vector


def get_cid_from_expression(expression):
    doc = firestore_client.collection('tesauro').select(field_paths=['`CID10_Código1`', '`CID10 mais frequente`']).where('`Termo Português`','==', expression).get()[0]
    return list(doc.to_dict().values())


def get_cid_title(cids):
    results = []
    for cid in cids:
        res = firestore_client.collection('cid').document(cid).get().to_dict()
        if res: results.append(res)
    return results