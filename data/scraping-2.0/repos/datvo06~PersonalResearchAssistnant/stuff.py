from langchain.document_loaders import UnstructuredFileLoader, TextLoader
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain import OpenAI
from langchain.prompts import PromptTemplate
import os
from langchain.callbacks import get_openai_callback
from settings import OPENAI_API_KEY

# Set OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

persistance_directory = 'db'
embeddings = OpenAIEmbeddings()

if not os.path.exists(persistance_directory):
    with open('book.txt', 'r', encoding='utf-8') as f:
        text = f.read().encode('utf-8', errors='ignore').decode('utf-8')
        with open('book_out.txt', 'w') as fo:
            fo.write(text)
    loader = TextLoader('book_out.txt')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    print("Embedding {} documents".format(len(docs)))
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistance_directory)
    db.persist()
else:
    db = Chroma(persist_directory=persistance_directory, embedding_function=embeddings)


# CUSTOM PROMPT
prompt_template = """Use the following pieces of context to answer the question at the end by summarizing the context. If you don't know the answer, just say that you don't know, don't try make up an answer

{context}

Question: {question}
Answer:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}


retriever=db.as_retriever()
# We are using the vectorstore as the database and not similarity searched docs as this is done in the chain.
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=db.as_retriever(), return_source_documents=False
                               , chain_type_kwargs=chain_type_kwargs)
#                                 )
if __name__ == '__main__':
    while True:
        with get_openai_callback() as cb:
            query = input("Enter query: ")
            result = qa({"query": query})
            print(result['result'])
            # print(result['source_documents'])
            print("tokens used: ", cb.total_tokens)
