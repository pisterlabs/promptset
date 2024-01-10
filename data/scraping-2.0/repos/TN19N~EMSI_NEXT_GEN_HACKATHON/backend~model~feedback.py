# Imports
from model.model import llm
from model.quiz import memory
from langchain import PromptTemplate
from langchain.chains import ConversationChain

# Template
template="""
The following is a feedback from the AI to human. The AI acts exactly like
a teacher, and the human acts exactly like a student. The goal is the teacher trys
to give a feedback on the quizs that the student take.

quizs:
{history}
AI:{input}""".strip()

prompt = PromptTemplate(input_variables=["input", "history"], template=template)

# Normaly Each memory and chain would unique to each user, (solution: use a database)
chain = ConversationChain(llm=llm, memory=memory, prompt=prompt, verbose=True)

def feedback() -> str:
    return chain.predict(input=None)