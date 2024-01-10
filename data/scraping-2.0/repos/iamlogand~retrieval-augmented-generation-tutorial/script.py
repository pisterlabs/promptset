import os
from pathlib import Path
from dotenv import load_dotenv
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

BASE_DIR = Path(__file__).resolve().parent
load_dotenv()

loader = PyPDFLoader(os.path.join(BASE_DIR, "src.pdf"))
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-4", temperature=0)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

chat_history = ""
while True:
    question = input("\n> ")
    chat_history += "\n" + question

    # Create a separate prompt for reformulating the question
    reformulation_template = (
        "Given the following conversation and a follow-up question, "
        "rephrase the follow-up question to be a standalone question, "
        "that will be used as the prompt for retrieving information from the instruction manual."
        "Chat History: {chat_history} "
        "Follow-up question: {question}"
    )
    reformulation_prompt = PromptTemplate(
        template=reformulation_template, input_variables=["chat_history", "question"]
    )
    reformulation_llm_chain = LLMChain(prompt=reformulation_prompt, llm=llm)

    reformulated_question = reformulation_llm_chain.run(
        {"chat_history": chat_history, "question": question}
    )

    answer = ""
    print()
    for chunk in rag_chain.stream(reformulated_question):
        answer += chunk
        print(chunk, end="", flush=True)
    chat_history += "\n" + answer
    print("")
