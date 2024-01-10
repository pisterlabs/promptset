from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import os

model = ChatOpenAI(temperature=0.5,
                   model_name="gpt-3.5-turbo-1106",
                   openai_api_key="API-KEY")

embeddings = OpenAIEmbeddings(openai_api_key="API-KEY")

script_directory = os.path.abspath(os.path.dirname(__file__))
file_path = os.path.join(script_directory, "context.txt")

loader = TextLoader(file_path,
                    encoding="utf8")

documents_lowsegment = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200,
                                               chunk_overlap=1)
texts = text_splitter.split_documents(documents_lowsegment)


vector_store = Chroma.from_documents(texts, embeddings)

template = """Sen aşağıdaki içeriğe göre kullanıcılara bilgi veren bir asistansın. Lütfen bilgin olmayan sorulara cevap vermeye çalışma.
İçerik: {texts}
Soru: {question}
Cevap: 
"""

qa = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

def query(q):
    prompt = PromptTemplate(input_variables=["question", "texts"],
                            template=template)

    prompt_format = prompt.format(question=q,
                                  texts=texts)
    return qa.run(prompt_format)


