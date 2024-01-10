# Imports
from model.model import llm
from model.quiz import memory
from langchain import PromptTemplate
from langchain.chains import ConversationChain

# Template
template="""
The following is a recommendation from the AI to human. The AI acts exactly like
a teacher, and the human acts exactly like a student. The goal is that the teacher try
to give a recommendations to the student on the quizs that the student take.

recommendations soulde be in this format:
topics to study: topic1, topic2, topic3
links for tutorials: link1, link2, link3

quizs:
{history}
AI:{input}""".strip()

prompt = PromptTemplate(input_variables=["input", "history"], template=template) 

# Normaly Each memory and chain would unique to each user, (solution: use a database)
chain = ConversationChain(llm=llm, memory=memory, prompt=prompt, erbose=True)

def recommendation() -> str:
    return chain.predict(input=None)