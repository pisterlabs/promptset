from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks import get_openai_callback
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader


memoria_chat_DOC =[]  #lista que guarda a memória do chat, dictionary de 3 keys: contexto do doc, input do user e resposta da IA
loader = TextLoader(file_path="MainIntegration/ferramentas.txt", encoding="utf-8", autodetect_encoding = False)
documents = loader.load()
llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature=0) # tentar mudar temperatura para testes.

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200, #cada chunk de texto tem no máximo 1600 caracteres, melhor numero que eu testei ate agora
    chunk_overlap  = 0,
 
     #divide partes do texto ao encontrar essa string, agrupa pedaços de texto entre a occorrencia dessa string
    
)

texts = text_splitter.split_documents(documents)
#print(texts)
embeddings = OpenAIEmbeddings()

docsearch = Chroma.from_documents(texts, embeddings)


def Pesquisa_Questoes(topic):
  
     docs = docsearch.similarity_search(topic, k=2)  #gera 1 pagina de resultado na busca por documentos
     #print(docs)
     inputs = [{"context": doc.page_content, "topic": topic} for doc in docs] #extrai um dictionary dos resultados da busca nos docs
     Quest_Docs = inputs[0]['context'] 
     print(" semantic search"+ Quest_Docs)
     Num1 = Quest_Docs.count("1")
     Num2 = Quest_Docs.count("2")
     Num3 = Quest_Docs.count("3")
     Num_mais_freq = max(Num1, Num2, Num3)
     if Num_mais_freq == Num1:
          return '1'
     elif Num_mais_freq ==Num2:
          return '2'
     elif Num_mais_freq ==Num3:
          return '3'

#Pesquisa_Questoes("a resposta dessa questão é a letra A?")