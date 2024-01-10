from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

def create_prompt(history):
    system_message = SystemMessage(content="""You are LangchainBot, a helpful learning assistant for people trying to learn the Python library Langchain.\n
    Respond in markdown to promote readability. Ensure your answers are complete and comprehensive, and also make sure to be concise. \n
    You can respond in any format that makes the most sense, including sentences, code blocks, bullet points or numbered lists depending on the question.\n
    Be friendly and informative. Speak simply, like a teacher. Answer the question based only on the provided context. \n
    If there is not enough context to answer the question, say you do not know. 
    """)
   
    rag_template = HumanMessagePromptTemplate.from_template(""" Answer the user question at the very bottom of this message. 
    Use the folwlowing context to answer the question. If you do not know, say you do not know. \n\n
    Context: \n[

    {context}

    ]\n""" + """\nThe following is the history of the conversation so far: \n [""" + 
    history + "]\n\n Please return your response in markdown. Be concise and speak in simple terms. \n user: \n<{question}>")

    prompt = ChatPromptTemplate.from_messages(
        [
            system_message,
            rag_template,
         
        ]
    )
    return prompt
