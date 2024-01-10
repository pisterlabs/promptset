import sys
sys.path.append('packages/langchain_')

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain,RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory,ConversationBufferWindowMemory

from llm_ import ChatGlmLLM


def test_base_chain():
    prompt_template = """基于以下已知信息,请简洁并专业地回答用户的问题,问题:{question}"""
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["question"])
    llm = ChatGlmLLM()
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run({"question":'你好'})
    print(result)


def memory_chain():
    template = """You are a chatbot having a conversation with a human.

    {chat_history}
    Human: {human_input}
    Chatbot:"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"], template=template)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    llm_chain = LLMChain(
        llm=ChatGlmLLM(),
        prompt=prompt,
        # verbose=True,
        memory=memory)
    llm_chain.predict(human_input="Hi there my friend 11")
    llm_chain.predict(human_input="Hi there my friend 22")
    llm_chain.predict(human_input="Hi there my friend 33")

    print(memory.chat_memory.messages)


def vectordb_chain():
    embedding_model_name='packages/langchain_/models/shibing624_text2vec-base-chinese'
    persist_directory = 'packages/langchain_/vectordb'

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vectordb = Chroma(embedding_function=embeddings,
                    persist_directory=persist_directory)
    
    llm = ChatGlmLLM()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=vectordb.as_retriever(),
        return_source_documents=True)

    query = "中央主题教育工作会议什么时候召开的"
    result = qa(query)
    print(result['source_documents'])


# test_base_chain()
memory_chain()
# vectordb_chain()
