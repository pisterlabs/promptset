from typing import Any, List
import concurrent.futures
import json
import pandas as pd
import requests
import streamlit as st
from google.protobuf.struct_pb2 import Struct

from clarifai.client.auth.helper import ClarifaiAuthHelper
from clarifai.client import create_stub
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

from langchain.chains import AnalyzeDocumentChain, ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings import ClarifaiEmbeddings
from langchain.llms import Clarifai as ClarifaiLLMs
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Clarifai as ClarifaiDB
from langchain.vectorstores import FAISS

USER_ID = "openai"
APP_ID = "chat-completion"
MODEL_ID = "GPT-3_5-turbo"

EMBED_USER_ID = "openai"
EMBED_APP_ID = "embed"
EMBED_MODEL_ID = "text-embedding-ada"


def get_search_query_text():
  input_text = st.text_input(
      "Search Query: ",
      placeholder=f"Type words or sentences to search for in your app",
      key="input_clqa",
      help="This will search for similar text in your Clarifai app.",
  )
  return input_text


def is_success(response):
  if response.status.code != status_code_pb2.SUCCESS:
    return False
  return True


def is_mixed_success(response):
  if response.status.code != status_code_pb2.MIXED_STATUS:
    return False
  return True


def calculate_expected_batch_number(inputs: List[Any], batch_size: int) -> int:
  """Function to calculate the expected number of batches

    Args:
        inputs (List[Any]): List of inputs
        batch_size (int): The number of items in the batch

    Returns:
        int: expected batch number
    """
  expected_batch_nums = ((len(inputs) // batch_size)
                         if len(inputs) % batch_size == 0 else (len(inputs) // batch_size + 1))
  return expected_batch_nums


def url_to_text(auth, url):
  try:
    h = {"Authorization": f"Key {auth.pat}"}
    response = requests.get(url, headers=h)
    response.encoding = response.apparent_encoding
  except Exception as e:
    print(f"Error: {e}")
    response = None
  return response.text if response else ""


def search_with_metadata(stub, userDataObject, search_metadata_key, search_metadata_value):
  search_metadata = Struct()
  search_metadata.update({search_metadata_key: search_metadata_value})
  post_searches_response = stub.PostSearches(
      service_pb2.PostSearchesRequest(
          user_app_id=userDataObject,
          query=resources_pb2.Query(ands=[
              resources_pb2.And(input=resources_pb2.Input(data=resources_pb2.Data(
                  metadata=search_metadata)))
          ]),
      ),)

  if post_searches_response.status.code != status_code_pb2.SUCCESS:
    print(post_searches_response.status)
    raise st.error(
        Exception("Post searches failed, status: " + post_searches_response.status.description))
  else:
    return post_searches_response


def process_post_searches_response(auth, post_searches_response):
  input_success_status = {
      status_code_pb2.INPUT_DOWNLOAD_SUCCESS,
      status_code_pb2.INPUT_DOWNLOAD_PENDING,
      status_code_pb2.INPUT_DOWNLOAD_IN_PROGRESS,
  }

  input_dict_list = []
  for idx, hit in enumerate(post_searches_response.hits):
    input = hit.input
    if input.status.code not in input_success_status:
      continue

    # Initializations
    input_dict = {}

    input_dict["input_id"] = input.id
    input_dict["text"] = url_to_text(auth, input.data.text.url)
    input_dict["source"] = input.data.metadata["source"]
    input_dict["text_length"] = input.data.metadata["text_length"]
    input_dict["page_number"] = input.data.metadata["page_number"]
    input_dict["page_chunk_number"] = input.data.metadata["page_chunk_number"]
    input_dict_list.append(input_dict)

  return input_dict_list


@st.cache_data(persist=True)
def get_clarifai_docsearch(user_input, number_of_docs, cache_id):
  auth = ClarifaiAuthHelper.from_streamlit(st)
  stub = create_stub(auth)
  userDataObject = auth.get_user_app_id_proto()

  docsearch = ClarifaiDB(
      user_id=userDataObject.user_id,
      app_id=userDataObject.app_id,
      pat=auth._pat,
      number_of_docs=number_of_docs*2)

  print("Searching for: %s" % user_input)
  docs = docsearch.similarity_search(user_input)
  docs = get_unique_docs(docs)
  return docs
  

def get_unique_docs(docs):
  unq_docs_meta = []
  unq_docs = []

  for doc in docs:
    if doc.metadata in unq_docs_meta:
      continue
    else:
      unq_docs_meta.append(doc.metadata)
      unq_docs.append(doc)

  return unq_docs


# Function to load custom llm chain
def load_custom_llm_chain(prompt_template):
  auth = ClarifaiAuthHelper.from_streamlit(st)
  pat = auth._pat
  llm_chatgpt = ClarifaiLLMs(pat=pat, user_id=USER_ID, app_id=APP_ID, model_id=MODEL_ID)
  prompt = PromptTemplate(template=prompt_template, input_variables=["page_content"])
  llm_chain = LLMChain(prompt=prompt, llm=llm_chatgpt)
  return llm_chain


@st.cache_data(
    persist=True, hash_funcs={
        list: lambda l: "".join([str(x.page_content) for x in l])
    })
def get_embeddings(pat, documents, cache_id):
  embedder = ClarifaiEmbeddings(
      pat=pat, user_id=EMBED_USER_ID, app_id=EMBED_APP_ID, model_id=EMBED_MODEL_ID)
  texts = [doc.page_content for doc in documents]
  embeddings = embedder.embed_documents(texts)
  return embeddings


def create_retrieval_qa_chat_chain(split_texts, cache_id):
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
  documents = text_splitter.create_documents(split_texts)
  auth = ClarifaiAuthHelper.from_streamlit(st)
  pat = auth._pat
  texts = [doc.page_content for doc in documents]
  embedder = ClarifaiEmbeddings(
      pat=pat, user_id=EMBED_USER_ID, app_id=EMBED_APP_ID, model_id=EMBED_MODEL_ID)
  embeddings = get_embeddings(pat, documents, cache_id)
  blah = zip(texts, embeddings)

  vectorstore = FAISS.from_embeddings([a for a in blah], embedder)

  memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False)
  retrieval_qa_chat_chain = ConversationalRetrievalChain.from_llm(
      ClarifaiLLMs(pat=pat, user_id=USER_ID, app_id=APP_ID, model_id=MODEL_ID),
      vectorstore.as_retriever(search_kwargs={'k': 3}),
      memory=memory,
      chain_type="stuff",
      return_source_documents=False,
      get_chat_history=lambda h: h)
  return retrieval_qa_chat_chain


# Function that gets the texts and stitches them to create a full text from Clarifai app
@st.cache_data(persist=True, hash_funcs={list: lambda l: "".join([str(x) for x in l])})
def get_full_text(docs, doc_selection, cache_id):
  auth = ClarifaiAuthHelper.from_streamlit(st)
  stub = create_stub(auth)
  userDataObject = auth.get_user_app_id_proto()
  meta_source = docs[doc_selection].metadata["source"] if "source" in docs[doc_selection].metadata.keys() else ""
  print("Searching for: %s" % meta_source)
  post_searches_response = search_with_metadata(
      stub,
      userDataObject,
      search_metadata_key="source",
      search_metadata_value=meta_source,
  )
  search_input_df = pd.DataFrame(process_post_searches_response(auth, post_searches_response))
  if 'page_number' and 'page_chunk_number' in search_input_df.columns:
    search_input_df = search_input_df.sort_values(["page_number", "page_chunk_number"])
  search_input_df.reset_index(drop=True, inplace=True)
  # st.session_state.search_input_df = search_input_df
  full_text = "\n".join(search_input_df.text.to_list())
  return full_text, search_input_df


# Function that gets summarization output using LLM chain
@st.cache_data(persist=True)
def get_summarization_output(full_text, cache_id):
  auth = ClarifaiAuthHelper.from_streamlit(st)
  pat = auth._pat
  llm = ClarifaiLLMs(pat=pat, user_id=USER_ID, app_id=APP_ID, model_id=MODEL_ID)
  summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
  summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)
  text_summary = summarize_document_chain.run(full_text)
  return text_summary


def combine_dicts(list_of_dicts):
  result = {}
  for dict_ in list_of_dicts:
    for key, value in dict_.items():
      result.setdefault(key, set()).update(value)
  return {k: sorted(list(v)) for k, v in result.items()}


def get_openai_output(_llm_chain, input):
  chat_output = _llm_chain(input)
  return chat_output


@st.cache_data(persist=True)
def extract_entities(llm_output):
  if isinstance(llm_output, dict) and len(llm_output) == 6:
    return llm_output
  elif isinstance(llm_output, str):
    entity_dict = {}
    llm_output = llm_output.lower().strip()
    if "output" in llm_output[:20]:
      llm_output = llm_output.split("output:")[1].strip()
    try:
      entity_dict = json.loads(llm_output)
    except Exception as e:
      st.error(f"output: {llm_output}\nerror: {e}")
    return entity_dict


@st.cache_data(persist=True)
def process_input(_llm_chain, input):
  chat_output = get_openai_output(_llm_chain, input["page_content"])
  entities = extract_entities(chat_output["text"])

  if entities == {}:
    return entities
  else:
    entities["SOURCES"] = [input["source"]] if 'source' in input.keys() else {}
  return entities


@st.cache_data(persist=True)
def parallel_process_input(_llm_chain, input_list):
  entities_list = []
  with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    future_entities = [executor.submit(process_input, _llm_chain, input) for input in input_list]
    for future in concurrent.futures.as_completed(future_entities):
      entities = future.result()
      entities_list.append(entities)
  return entities_list
