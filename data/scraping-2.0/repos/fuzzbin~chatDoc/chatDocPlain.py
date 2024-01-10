# Prototype for å spørre mot et pdf-dokument

# Importerer nødvendige biblioteker
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
import logging
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import pprint as pp
import dotenv

dotenv.load_dotenv() # Henter miljøvariabler fra .env-fil

# Variabler
document_path = "./documents/orden.pdf"

# Laster inn dokumentet
loader = UnstructuredPDFLoader(document_path, mode="single", strategy="fast")
doc = loader.load()

# Splitter og lager en vectorstore av dokumentet
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
all_splits = text_splitter.split_documents(doc)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# Løkke som lar brukeren stille spørsmål til dokumentet
while True:
    question = input("Tast inn ditt spørsmål: ")
    docs = vectorstore.similarity_search(question)

    # Enkel logging
    logging.basicConfig()
    logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

    # Lager tre ulike varianter av spørsmålet og henter ut relevnate deler fra dokumentet (Prompt-tuning)
    retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=ChatOpenAI(temperature=0))
    unique_docs = retriever_from_llm.get_relevant_documents(query=question)

    # Valg av modell
    llm1 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    llm2 = HuggingFaceHub(repo_id="RuterNorway/Llama-2-13b-chat-norwegian-GPTQ", model_kwargs={"temperature":0.1, "max_new_tokens":250})

    # Bytt ut llm1 med llm2 for å bruke en annen modell
    qa_chain = RetrievalQA.from_chain_type(llm1,retriever=vectorstore.as_retriever())
    pp.pprint(qa_chain({"query": question}))
    print("\n")