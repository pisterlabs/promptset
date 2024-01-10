import streamlit as st
from langchain.llms import OpenAI
import logging
import sys


st.title('Fruitstand Support App')

from typing import List
from typing import Dict
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

import json

from langchain.docstore.document import Document

FAISS_INDEX_DIR = "faiss_index"

class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, inputs: list[str], model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"inputs": inputs, **model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> List[List[float]]:
        response_json = json.loads(output.read().decode("utf-8"))
#        return response_json["vectors"]
        return response_json[0]["generated_text"]


logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
file_handler = logging.FileHandler('queries.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)


logger.addHandler(file_handler)

content_handler = ContentHandler()

embeddings = HuggingFaceEmbeddings()
new_db = FAISS.load_local(FAISS_INDEX_DIR, embeddings)

def generate_response(input_text):
  llm2 = SagemakerEndpoint(
    endpoint_name="hf-llm-falcon-40b-instruct-bf16-2023-06-23-19-34-33-102",
    #endpoint_name="hf-llm-falcon-7b-bf16-2023-06-24-20-08-14-262",
    #endpoint_name="hf-llm-falcon-40b-bf16-2023-06-24-20-20-44-608",
    model_kwargs={
         "parameters" : {"do_sample": False,
        "top_p": 0.9,
        "temperature": 0.1,
        "max_new_tokens": 400
                  }},
    region_name="us-east-1",
    content_handler=content_handler
  )
  
  llm_query= input_text
  #logger.info(input_text)

  prompt_template = """
  {context}
  >>QUESTION<<: {question}
  >>ANSWER<<:"""
  
  PROMPT = PromptTemplate(
      template=prompt_template, input_variables=["context", "question"]
  )
  
  chain = load_qa_chain(llm=llm2, prompt=PROMPT)
  
  docs = [Document(page_content=" ",    )]
  docs = new_db.similarity_search(llm_query)
  
  #logger.info(docs)
  
  
  output = chain({"input_documents":docs, "question": llm_query}, return_only_outputs=False)
  logger.info(output)
  st.info(output['output_text'])

with st.form('my_form'):
  text = st.text_area('Enter your query:', 'How do I charge my iPhone?')
  submitted = st.form_submit_button('Submit')
  if submitted:
    generate_response(text)
    

with st.sidebar:
  add_markdown= st.subheader('About the demo')
  add_markdown= st.markdown('This is a sample application that uses **Falcon 40b Instruct** with RAG using FAISS. Data for RAG is from the Apple support pages')
  add_markdown= st.markdown('You can ask questions like **:blue["my iphone screen is broken, how can I fix it"]** or **:blue["how do I change the wallpaper on my iphone"]**')
  
