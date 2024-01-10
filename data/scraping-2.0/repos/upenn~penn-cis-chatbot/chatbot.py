import sys
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from persistence import Persistence
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
If there are two or three likely answers, list all of the likely answers.
Use three sentences maximum and keep the answer as concise as possible. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", 
                 temperature=0)

vectordb = Persistence.get_storage('index_these')

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    #verbose=True,
    memory=ConversationBufferWindowMemory(k=2),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        result = qa_chain({"query": question})
        print(result["result"])


