from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain

from langchain.schema import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import time
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
def make_chain():
    QA_PROMPT_DOCUMENT_CHAT = """You're name is Aziz you are a university assistant,
    help answer student questions about the university, you can answer in the language the question is asked in. 
    Use the following pieces of context to answer the question at the end.
    If the question is not related to the context, politely respond that you 
    are teached to only answer questions that are related to the context.
    If you're asked about your name or idendity please answer with your name and your role.
    If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
    Answer in the language you are asked in.
    {context}

    Question: {question}
    Helpful Answer:"""

    # Create a PromptTemplate with your custom template
    custom_condense_question_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=QA_PROMPT_DOCUMENT_CHAT
    )
    model = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature="0.4",
        


        # verbose=True
    )
    embedding = OpenAIEmbeddings()

    vector_store = Chroma(
        collection_name="CNPN-Licence-Fondamentale_Professionnelle",
        embedding_function=embedding,
        persist_directory="app/model/src/data/chroma",
    )

    return ConversationalRetrievalChain.from_llm(
        model,
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": custom_condense_question_prompt},
        condense_question_prompt=custom_condense_question_prompt
        # verbose=True, 
    )


def gen_answer(user_input,chat_history_input=None):
    load_dotenv()

    chain = make_chain()
    chat_history = []

    if chat_history_input:
       for chat in chat_history_input['chat history']:
              if(list(chat_history_input['chat history']).index(chat)%2==0):
               print(chat['content'])  
               chat_history.append(HumanMessage(content=chat['content']))
              else:
               chat_history.append(AIMessage(content=chat['content']))
           
    else:
       chat_history = []

    print('okey',chat_history)
    question = user_input

    # Generate answer
    response = chain({"question": question, "chat_history": chat_history})

    # Retrieve answer
    answer = response["answer"]
    
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))
    return answer,chat_history
   


print(gen_answer("goliya bdarija chnahiya license fondamentale")[0])