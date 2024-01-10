import os
import sys
import json
import requests
from langchain.chains import ConversationalRetrievalChain, RetrievalQAWithSourcesChain, RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.memory import ConversationBufferMemory
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAIChat
from langchain.vectorstores import Chroma
from langchain.schema import SystemMessage, HumanMessage
from langchain.vectorstores import FAISS
import logging
from prompts.default import vector_store_prompt
from time import sleep
from custom_encodings.helper import count_tokens
from validation import process_html_with_selectors

logging.getLogger("openai").setLevel(logging.DEBUG)
os.environ['OPENAI_API_KEY'] = "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"
gpt_model = "gpt-4"

if len(sys.argv) > 1:
  query = sys.argv[1]

messages = {}
with open('prompts/prompts.json') as prompts:
    prompts = json.load(prompts)

raw_documents = TextLoader('data/data.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50, length_function=count_tokens)
documents = text_splitter.split_documents(raw_documents)

embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(documents, embeddings)

search_result = vector_store.similarity_search_with_score("what is the css selector for text - 'Carbonated Water, Sugar, Caramel, Phosphoric Acid, Flavouring and Caffeine' in the html", k=5)
print(search_result)

# for field in prompts:
#   search_result = vector_store.similarity_search_with_score(prompts[field], k=5)
#   print(search_result)

# system_template={"role": "assistant", "content": "You are an assistant that has all the knowledge about CSS selectors and web scraping."}
# messages.append(system_template)

# system_template={"role": "assistant", "content": "You are an assistant that has all the knowledge about CSS selectors and web scraping."}
# messages.append(system_template)

# chain = ConversationalRetrievalChain.from_llm(
#             llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"), 
#             retriever=vector_store.as_retriever(),
#             )

# chain = ConversationalRetrievalChain.from_llm(
#             llm=OpenAI(temperature=0, model="gpt-4"), 
#             chain_type="stuff",
#             retriever=vector_store.as_retriever(),
#             model_kwargs={"memory":ConversationBufferMemory()}
#             )

for field in prompts:
  retriever = vector_store.as_retriever(search_kwargs={"k": 5}, search_type='similarity')
  chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0, model=gpt_model),
                                    chain_type="map_rerank",
                                    retriever=retriever,
                                    return_source_documents=True,
                                    verbose=False)
  # prompt = f'<>{vector_store.similarity_search(prompts[field], k=2)}<> {get_vector_store_prompt(prompts[field])}'
  prompt = prompts[field] + vector_store_prompt
  # result = chain({"question": prompt, "chat_history": messages})
  try:
    result = chain(prompt)
  except Exception as e:
    print("length exceeded! Retrying with smaller length")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3, "max_length": 100}, search_type='similarity')
    chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0, model=gpt_model),
                                      chain_type="map_rerank",
                                      retriever=retriever,
                                      return_source_documents=True,
                                      verbose=False)
    # prompt = f'<>{vector_store.similarity_search(prompts[field], k=2)}<> {get_vector_store_prompt(prompts[field])}'
    prompt = prompts[field] + vector_store_prompt
    # result = chain({"question": prompt, "chat_history": messages})

  error_list = ["does not contain any information", "not explicitly provided in the given context.", "not", "N/A", "no"]
  for error in error_list:
     if error in result['result']:
      print("recheking")
      retriever = vector_store.as_retriever(search_kwargs={"k": 5}, search_type='similarity')
      chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0, model=gpt_model),
                                    chain_type="stuff",
                                    retriever=retriever,
                                    return_source_documents=True,
                                    verbose=False)
      # prompt = f'<>{vector_store.similarity_search(prompts[field], k=2)}<> {get_vector_store_prompt(prompts[field])}'
      prompt = prompts[field] + vector_store_prompt
      # result = chain({"question": prompt, "chat_history": messages})
      try:
        result = chain(prompt)
      except Exception as e:
        print("length exceeded! Retrying with smaller length")
        retriever = vector_store.as_retriever(search_kwargs={"k": 3, "max_length": 100}, search_type='similarity')
        chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0, model=gpt_model),
                                          chain_type="stuff",
                                          retriever=retriever,
                                          return_source_documents=True,
                                          verbose=False)
        # prompt = f'<>{vector_store.similarity_search(prompts[field], k=2)}<> {get_vector_store_prompt(prompts[field])}'
        prompt = prompts[field] + vector_store_prompt
        # result = chain({"question": prompt, "chat_history": messages})


  for error in error_list:
    if error not in result['result']:
        messages[field] = result['result']
    else:
      messages[field] = "N/A"

  print(result['result'])
  chain = None
  retriever = None
  sleep(5)

json_string = json.dumps(messages)
with open('data/output.json', 'w') as file:
    file.write(json_string)

# validation_messages = process_html_with_selectors('data/data.txt', messages)
# json_string = json.dumps(validation_messages)
# with open('data/validation_output.json', 'w') as file:
#     file.write(json_string)

# while True:
#   query = None
#   if not query:
#     query = input("Prompt: ")
#   if prompt in ['quit', 'q', 'exit']:
#     sys.exit()

#   prompt = HumanMessage(f' {vector_store.similarity_search(query)} {vector_store_prompt(query)}')
#   # result = chain({"question": prompt, "chat_history": messages})
#   messages.append((prompt, result['result']))
#   retriever = vector_store.as_retriever(search_kwargs={"k": 5})
#   chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0, model="gpt-4"),
#                                     chain_type="stuff",
#                                     retriever=retriever,
#                                     return_source_documents=True,
#                                     verbose=False)
#   result = chain(prompt)
#   print(result['result'])
#   query = None