from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

template = """
You are a helpful AI chatbot on Colin's website who is having a conversation with a human. Use the following pieces of context about Colin to reply to the human's message.
If the human is asking a question, and you don't know the answer, just say you don't know. DO NOT try to make up an answer.

{context}

Message: {question}
Reply:"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])

def get_chain(vectorstore):
    llm = ChatOpenAI(temperature=0.5,model_name='gpt-3.5-turbo')
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
        combine_docs_chain_kwargs = {'prompt':QA_PROMPT},
    )
    return qa_chain