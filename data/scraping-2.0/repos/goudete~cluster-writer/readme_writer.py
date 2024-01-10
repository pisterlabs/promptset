import os

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings


from langchain.chains import RetrievalQA

# Keys
os.environ['OPENAI_API_KEY'] = ""
# os.environ["ACTIVELOOP_TOKEN"] = ""

'''
Data Loader

Extracts custom data from a directory and loads it into a list of Document objects.
'''


'''
Data Transformer

Transforms documents into smaller chunks that can fit into the models context window.
When dealing with long pieces of text, it is often necessary to split the text into chunks.

At a high level, text splitters work as following:

    - Split the text up into small, semantically meaningful chunks (often sentences).
    - Start combining these small chunks into a larger chunk until you reach a certain size (as measured by some function).
    - Once you reach that size, make that chunk its own piece of text and then start creating a new chunk of text with some overlap (to keep context between chunks).
    - That means there are two different axes along which you can customize your text splitter:
        - How the text is split
        - How the chunk size is measured

Notes:
For my use case, I need to split documents by class and function groups.

Reference:
Context aware text splitting: https://python.langchain.com/docs/use_cases/question_answering/how_to/document-context-aware-QA

'''
# print("Transforming data...")
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 200,
#     chunk_overlap  = 20,
#     length_function = len,
# )
# all_splits = text_splitter.create_documents([
#     file.page_content for file in files
# ])


'''
Data Retriever

A retriever is an interface that returns documents given an unstructured query.
It is more general than a vector store. A retriever does not need to be able to 
store documents, only to return (or retrieve) it. Vector stores can be used as 
the backbone of a retriever, but there are other types of retrievers as well.

'''
# print("Creating vectorstore...")
# vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# question = "What is the provided context?"

# qa_chain = RetrievalQA.from_chain_type(
#     llm,
#     retriever=vectorstore.as_retriever(),
# )

# result = qa_chain.run(question)

# print(result)
