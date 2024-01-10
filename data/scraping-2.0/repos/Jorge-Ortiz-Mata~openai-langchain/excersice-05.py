# Transcript videos from Youtube

# Install youtube-transcript-api, tiktoken and faiss-gpu packages

from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

# This method create the embedings [0.2, 0.9, -0.2, ...] using the FAISS vector store
def youtube_vector(url: str) -> FAISS:
  loader = YoutubeLoader.from_youtube_url(url)
  transcript = loader.load()

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
  docs = text_splitter.split_documents(transcript)

  db = FAISS.from_documents(docs, embeddings)

  return db

# This method sends back the response base on the vector store from the previous method and the query (question) made
def ask_something(db, query, k=4):
  # Do somenthing

  docs = db.similarity_search(query, k=k)
  docs_page_content = " ".join([d.page_content for d in docs])

  llm = ChatOpenAI(temperature=0.6, model_name="gpt-3.5-turbo")

  prompt_template = PromptTemplate(
    input_variables=["question", "docs"],
    template="""
      Hello, please help me with this question: {question}
      This is the video url where you can get the info: {docs}

      If you do not have the answer, please say I do not know
    """
  )

  chain = LLMChain(llm=llm, prompt=prompt_template)

  response = chain.run(question=query, docs=docs_page_content)

  response.replace("\n", "")

  return response


video_url='https://youtu.be/K_aShV1p_EI?si=LfDX3wh8uED-aHgv'
query = '¿What are they talking about?'

youtube_video_vector = youtube_vector(video_url) # Get the vector stores
response = ask_something(youtube_video_vector, query) # Send and get the response from ChatGPT

print(response) # print response

