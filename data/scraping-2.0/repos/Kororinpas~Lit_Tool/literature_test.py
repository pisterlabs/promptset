import streamlit as st
import sys


class StreamlitWriter:
    def write(self, text):
        st.write(text.strip())

### This the function about streamlit
def Vector_Databse():
    st.write("Vector Database")
    choose = st.radio("Choose using an existing database or upload a new one.",
                      ["Using an existing one", "Uploading a new one"])
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if choose == "Using an existing one":
        persist_dirctory = st.text_input("Enter the persist_dirctory")
        collection = st.text_input("Enter the collection")

        if st.button('Confirm'):
            st.session_state['persist_dirctory'] = persist_dirctory
            st.session_state['collection'] = collection

            vectorstore,embeddings = load_vectorstore(persist_directory=st.session_state['persist_dirctory'],
                                           collection_name = st.session_state['collection'],
                                           model_name = 'sentence-transformers/all-mpnet-base-v2',
                                           device = device)

            st.session_state['vectorstore'] = vectorstore
            st.session_state['embeddings'] = embeddings
            print('The vectorstore load successfully')


    else:
        path = st.text_input("Enter the path")
        persist_dirctory = st.text_input("Enter the persist_dirctory")
        collection = st.text_input("Enter the collection")

        if st.button('Confirm'):
            st.session_state['path'] = path
            st.session_state['persist_dirctory'] = persist_dirctory
            st.session_state['collection'] = collection

            split_docs = load_pdf(path = st.session_state['path'],
                                  openai_api_key=st.session_state['openai_api_key'],
                                  chunk_size=st.session_state['chunk_size'],
                                  chunk_overlap=st.session_state['chunk_overlap'])

            vectorstore,embeddings = generate_vectorstore(split_docs = split_docs,
                                               model_name = 'sentence-transformers/all-mpnet-base-v2',
                                               persist_directory = st.session_state['persist_dirctory'],
                                               collection_name = st.session_state['collection'],
                                               device=device)

            st.session_state['vectorstore'] = vectorstore
            st.session_state['embeddings'] =embeddings
            print('The vectorstore load successfully')


def Parameters():
    import os
    openai_api_key = st.text_input('Enter your Openapi_api_key')
    if st.button('Confirm'):
        if openai_api_key == '':
            st.session_state['openai_api_key'] = os.environ.get('openai_api_key')
        else:
            st.session_state['openai_api_key'] = openai_api_key

    chunk_size = st.text_input('Enter your chunk_size')
    if st.button('Confirm_1'):
        if chunk_size== '':
            st.session_state['chunk_size'] = 1500

    chunk_overlap = st.text_input('Enter your chunk_overlap')
    if st.button('Confirm_2'):
        if chunk_overlap == '':
            st.session_state['chunk_overlap'] = 0




def Docs():
    col1,col2 = st.columns([1,1])
    with col1:
        output_text = ''
        vectorstore = st.session_state['vectorstore']
        edited_output_text = st.text_area("输出文本", value=output_text, height=600)
        if st.button("Confirm paragraph"):
            output_text = edited_output_text

        k = st.slider("Select the number of sentences to generate", min_value=1, max_value=5, value=1)

        query = st.text_input("Input the query")
        if st.button("Confirm query"):
            output, docs = get_chain_output(query=query,
                                            vectordb=vectorstore,
                                            k=k,
                                            openai_api_key=st.session_state['openai_api_key'])
            final_json = run_text_match(output=output,
                                        query=query,
                                        docs=docs,
                                        k=k,
                                        embeddings=st.session_state['embeddings'])
            st.session_state['final_json'] = final_json

    with col2:
        if 'final_json' in st.session_state:
            final_json = st.session_state['final_json']
            selected_sentence = st.selectbox("Select a sentence", final_json)
            if st.button('Confirm sentence'):
                process_selected_sentence(selected_sentence)






###This is the function about Langchain

###Loading PDF part
def load_pdf(path, openai_api_key, chunk_size, chunk_overlap):
    from langchain.document_loaders import PyMuPDFLoader, DirectoryLoader, UnstructuredPDFLoader
    #from detectron2.config import get_cfg
    from PyPDF2 import PdfReader

    #cfg = get_cfg()
    #cfg.MODEL.DEVICE = 'gpu'

    import os

    file_names = os.listdir(path)
    pdf_file_names = [path + '/' + file_name for file_name in file_names if file_name.endswith('.pdf')]

    docs = []

    import re

    for pdf in pdf_file_names:
        source = extract_doi(pdf)

        if source != 'None':
            doc = PyMuPDFLoader(pdf).load()
            for element in doc:
                element.metadata = source
                element.page_content = re.sub('\n+', ' ', element.page_content.strip())
                docs.append(element)

        else:
            doc = PyMuPDFLoader(pdf).load()
            print(f"{pdf} is not identified! Using other strategy!!")
            source = extract_doi_llm(doc, openai_api_key)
            if source != 'None':
                for element in doc:
                    element.metadata = source
            for element in doc:
                element.page_content = re.sub('\n+', ' ', element.page_content.strip())
                docs.append(element)

    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    split_docs = text_splitter.split_documents(docs)

    return split_docs


def get_info(path):
    from PyPDF2 import PdfReader
    with open(path, 'rb') as f:
        pdf = PdfReader(f)
        info = pdf.metadata
        return info


def extract_doi(path):
    source = 0
    info = get_info(path)
    if '/doi' in info:
        doi = info['/doi']
    elif '/Subject' in info:
        Subject = info['/Subject']
        if 'doi:' in Subject:
            Subject = Subject.split('doi:')
            doi = Subject[1]
        else:
            source = 'None'
    elif '/WPS-ARTICLEDOI' in info:
        doi = info['/WPS-ARTICLEDOI']
    else:
        source = 'None'

    if source != 'None':
        import habanero
        import time
        citation = habanero.cn.content_negotiation(ids=doi, format='bibentry')
        time.sleep(5)
        import bibtexparser
        citation = bibtexparser.loads(citation)
        citation = citation.entries[0]
        source = {'author': citation['author'],
                  'year': citation['year'],
                  'title': citation['title'],
                  'journal': citation['journal'],
                  }

    return source

def extract_doi_llm(doc,openai_api_key):

  import re

  doc[0].page_content = re.sub('\n+',' ',doc[0].page_content.strip())

  from langchain.text_splitter import RecursiveCharacterTextSplitter
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap = 50)
  split_docs = text_splitter.split_documents(doc)
  abstract = split_docs[0]
  doi = extract_chain(abstract,openai_api_key)

  if doi != 'None' and doi!= None:
    import habanero
    import time
    citation = habanero.cn.content_negotiation(ids = doi,format='bibentry')
    time.sleep(5)
    import bibtexparser
    citation = bibtexparser.loads(citation)
    citation = citation.entries[0]
    source = {'author':citation['author'],
            'year':citation['year'],
            'title':citation['title'],
            'journal':citation['journal'],
            }
    return source
  else:
    source = 'None'
    return source


def extract_chain(abstract, openai_api_key):
    from kor.extraction import create_extraction_chain
    from kor.nodes import Object, Text, Number
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=openai_api_key,
        temperature=0,
    )
    schema = Object(
        id="doi",
        description="doi is a digital identifier.It typically starts with 10. followed by a numeric prefix, such as 10.1000/182.",
        attributes=[
            Text(
                id="doi",
                description='doi is a digital identifier. It typically starts with "10." followed by a numeric prefix, such as 10.1000/182.',
                examples=[
                    (
                    'American Economic Journal: Economic Policy 2015, 7(4): 223–242  http://dx.doi.org/10.1257/pol.20130367 223 Water Pollution Progress at Borders: The',
                    'http://dx.doi.org/10.1257/pol.20130367'),
                    (
                    'Environment and Development Economics (2020), 1–17 doi:10.1017/S1355770X2000025X EDE RESEARCH ARTICLE Political incentives, Party Congress, and pollution cycle: empirical evidence from China Zhihua Tian,1 and Yanfang Tian2* 1School of Economics, Zhejiang University of Technology, Hangzhou',
                    '10.1017/S1355770X2000025X')
                ],
                many=True
            )
        ],
        many=False
    )
    chain = create_extraction_chain(llm, schema, encoder_or_encoder_class='json')
    output = chain.predict_and_parse(text=abstract.page_content)
    if 'doi' not in output['data']:
        print(f"LLM strategy failed!!{abstract.metadata['source']} Please manually add it!!")
        source = 'None'

        return source

    else:
        if output['data']['doi']['doi'] == []:
            print(f"LLM strategy failed!!{abstract.metadata['source']} Please manually add it!!")
            source = 'None'
            return source
        else:
            doi = output['data']['doi']['doi'][0]
            if 'doi=' in doi:
                doi = doi.split('doi=')[1]
                return doi

###Loading the database
def generate_vectorstore(split_docs, device, model_name, persist_directory, collection_name):
    from langchain.vectorstores import Chroma
    from langchain.embeddings import HuggingFaceEmbeddings

    model_kwargs = {'device': device}
    model_name = model_name
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    persist_directory = persist_directory
    collection_name = collection_name

    vectorstore = Chroma.from_documents(split_docs, embeddings, collection_name=collection_name,
                                            persist_directory=persist_directory)
    vectorstore.persist()

    return vectorstore,embeddings

def load_vectorstore(persist_directory,device,model_name,collection_name):
    from langchain.vectorstores import Chroma
    from langchain.embeddings import HuggingFaceEmbeddings

    model_kwargs = {'device': device}
    model_name = model_name
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    vectordb = Chroma(collection_name=collection_name,
                      persist_directory=persist_directory,
                      embedding_function=embeddings)

    return vectordb,embeddings

###Using Langchain and match

def get_chain_output(query, vectordb, k, openai_api_key):
    docs = vectordb.similarity_search(query, 6, include_metadata=True)


    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0, model_name="gpt-3.5-turbo")

    from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
    from langchain.llms import OpenAI

    from langchain.output_parsers import PydanticOutputParser
    from pydantic import BaseModel, Field, validator
    from typing import List, Union, Optional

    class Sentence(BaseModel):
        sentence: List[str] = Field(
            description="The sentence in the given document which is the most similar to the query provided")
        source: List[str] = Field(description="The meta source of the paper")
        score: List[float] = Field(
            description="The similarity score between the sentence selected and the query provided")

    parser = PydanticOutputParser(pydantic_object=Sentence)

    dic = {'1':"one",
           "2":"two",
           "3":"three",
           "4":"four",
           "5":"five"}
    k = dic[str(k)]

    question_template = f"""
    Given the document and query, find {k} sentences in the document that are most similar in meaning to the query. 
    Return the sentences, the meta source of the sentences and the cosine similarity scores. 
    If no similar sentences is found, return the sentence with highest cosine siliarity scores.
    """
    main_template = """
    {query}
    ===========
    {context}
    ===========
    {format_instructions}

    """
    question_template = question_template+main_template


    from langchain.chains.question_answering import load_qa_chain
    from langchain import LLMChain

    PROMPT = PromptTemplate(template=question_template,
                            input_variables=['query', 'context'],
                            partial_variables={"format_instructions": parser.get_format_instructions()})

    llm_chain = LLMChain(llm=llm, prompt=PROMPT)

    output = llm_chain({"query": query, "context": docs})

    return output, docs


def run_text_match(output, k,query, docs,embeddings):
    import re
    text = re.sub("\n+", "", output['text'])

    import json
    json_obj = json.loads(text)

    if "properties" in json_obj:
        print('No result was found, Using embedding searching strategy!!!')
        split_docs = split_for_embedding(docs)
        similar_sentence = search_cosine_similarity(query,k,split_docs, embeddings)

        return similar_sentence

    else:
        json_obj = [{'sentence': json_obj['sentence'][i],
                           'source': json_obj['source'][i],
                           'score': json_obj['score'][i]} for i in range(k)]
        return json_obj


def split_for_embedding(docs):  ##输入docs(list),输出split_for embedding(list)
    for_embedding = []
    for content in docs:
        new_content = content.page_content.replace('et al.', 'et al。')
        new_content = new_content.split('.')

        if 'source' in content.metadata:
            meta_data = content.metadata['source']
        else:
            meta_data = content.metadata

        for split_content in new_content:
            split_content = split_content.replace('。', '.')

            if len(split_content) < 30:
                continue
            else:
                for_embedding.append({"content": split_content, "source": meta_data})

    return for_embedding


def search_cosine_similarity(query, k,split_docs, embeddings):  ##query-str,split_docs-list,embeddings-embeddings()
    split_docs_content = [content['content'] for content in split_docs]

    split_docs_content = list(set(split_docs_content))

    embed_docs = embeddings.embed_documents(split_docs_content)
    embed_query = embeddings.embed_query(query)

    from openai.embeddings_utils import cosine_similarity

    cos_index = []
    for embed_doc in embed_docs:
        cos_index.append(cosine_similarity(embed_doc, embed_query))

    # 这边是根据大小建立索引
    idx = sorted(range(len(cos_index)), key=lambda k: cos_index[k])  # 根据cos_index的大小进行排序
    final_similar_list = []
    for index in idx[-k:]:
        unit = {}
        unit['sentences'] = split_docs_content[index]
        unit['source'] = split_docs[index]['source']
        unit['score'] = cos_index[index]
        final_similar_list.append(unit)

    return final_similar_list

def main():
  st.title("Literature Review Tool")
  sys.stdout = StreamlitWriter()

  # Create a toggle button to switch between pages
  page = st.sidebar.radio("Choose a page", [ "Parameter","Vector Database","Docs"])


  if page == "Parameter":
      Parameters()

  elif page == "Vector Database":
      Vector_Databse()

  elif page == "Docs":
      Docs()


def my_function(input_text):
    # 在此处添加您的处理逻辑
    output_text = input_text.upper()
    return output_text



def process_selected_sentence(selected_sentence):
    # 在最终输出区域展示用户选择的句子
    st.write(f"You selected: {selected_sentence}")

main()