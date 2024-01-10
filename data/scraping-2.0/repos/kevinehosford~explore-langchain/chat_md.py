from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

vectordb = Chroma(persist_directory='./md_vectors', embedding_function=OpenAIEmbeddings())

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever())

question = "How do you summarize the average of a field"

result = qa_chain({'query': question })

print(result['result'])
