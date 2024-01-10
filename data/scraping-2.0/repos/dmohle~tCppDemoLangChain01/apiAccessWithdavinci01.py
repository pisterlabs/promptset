import openai
import os
import sys

import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY


# Your OpenAI API key
# api_key = "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"

# Set up the client with your API key
 #openai.api_key = api_key

# Choose the model you want to use
# model_name = "text-davinci-003"  # As of my last update, GPT-3.5 model names may vary, please check OpenAI's documentation for the exact model name.

# The prompt you want to send to the model
# prompt_text = ("""A What year is it now? """
# )

# Make a request to the model
# try:
#  response = openai.Completion.create(
#    engine=model_name,
#   prompt=prompt_text,
#   max_tokens=1000
#  )

  # Print out the response
#  print(response.choices[0].text.strip())
#except openai.error.OpenAIError as e:
#  print(f"An error occurred: {e}")



from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain

from langchain.document_loaders import TextLoader
loader = TextLoader("notes06nov23.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=2400, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.4), vectorstore.as_retriever(), memory=memory)

query = "Is Hutson a good student? How is he doing in the class?"
result = qa({"question": query})

print("\n\n")
print(result)
print(type(result))
for key, value in result.items():
    print(f'{key}: {value}')
    print(f'{key}')
    if key == "answer":
        print(f'{value}')

