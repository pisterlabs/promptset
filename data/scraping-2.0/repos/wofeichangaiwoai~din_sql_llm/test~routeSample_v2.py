from datetime import datetime

from langchain import PromptTemplate, LLMChain
from tqdm import tqdm

from ubix.common.llm import get_llm

prompt = PromptTemplate(
    input_variables=["input"],
    template="""
You are currently doing a classification task, for question about\
data or table, classify them into Category '''query'''. For other type of questions, \
classify them into Category '''other'''. Your answer must be only one word, \
either '''query''' or '''other'''. Here are a few of examples: \

User: How many records in the table? \
Assistant: query \

User: What's the max number in table \
Assistant: query \

User: Could you help to summary how many sells in this quarter. \
Assistant: query \

User: who are you? \
Assistant: other \

User: what is your name? \
Assistant: other \

User:{input}
"""
)

search_chain = LLMChain(llm=get_llm(), prompt=prompt)

for _ in tqdm(range(1)):
    question_list = [
        "Hello, I'm Felix",
        # "Who are you?",
        # "How many records in this table",
        "What is the maximum total in this table?",
        "What is black body radiation?",
    ]
    for question in question_list:
        start = datetime.now()
        answer = search_chain.run(question)
        end = datetime.now()
        duration_route = (end-start).total_seconds()
        duration = (end-start).total_seconds()
        print(f">>>>>"*10 + f"\nEnd ask about question {question}, cost:{duration} sec, answer:{answer}\n" + "<<<<"*10)

"""
python test/routeSample.py
"""