from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

def create_prompt(history):
    system_message = SystemMessage(content="""You are ChemoBot, a chatbot that helps patients understand chemotherapy. Your job is to provide helpful, kind, and concise
    answers to patient's questions about various aspects of chemotherapy. If you do not know the answer to a question based on the context provided,
    say you do not know. If a patient is asking for medical advice, say you cannot provide medical advice and direct them to a medical professional.""")
    rag_template = HumanMessagePromptTemplate.from_template("""Answer the question based only on the following context. If there is not enough context to answer
                                                            the question, say you do not know.:

    {context}

    """)
                                                            
    chat_history_template = HumanMessagePromptTemplate.from_template("""The following is the history of the conversation so far: \n""" + history + "\n user: {question}")


    prompt = ChatPromptTemplate.from_messages(
        [
            system_message,
            rag_template,
            chat_history_template,
         
        ]
    )
    return prompt
