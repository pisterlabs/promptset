from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WikipediaLoader

import os
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

def answer_question_about(person_name,question):
    # Get Wikipedia Article
    docs = WikipediaLoader(query=person_name,load_max_docs=1)
    context_text = docs.load()[0].page_content
    
    # Connect to OpenAI Model
    model = ChatOpenAI()
    
    # Ask Model Question
    human_prompt = HumanMessagePromptTemplate.from_template('Answer this question\n{question}, here is some extra context:\n{document}')
    
    # Assemble chat prompt
    chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
    
    #result
    result = model(chat_prompt.format_prompt(question=question,document=context_text).to_messages())
    
    print(result.content)

answer_question_about("Claude Shannon","When was he born?")