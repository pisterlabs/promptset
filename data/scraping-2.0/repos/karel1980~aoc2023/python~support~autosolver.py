from collections import Counter
from langchain.chat_models import ChatOllama
from langchain.prompts import (
    ChatPromptTemplate,
)
from langchain.schema.output_parser import StrOutputParser
from langchain_experimental.utilities import PythonREPL

template = """Write a python function to solve the user's problem.

Your response must be in markdown format. All code must be between comment lines which indicate the beginning and the end of the code.

The problem description starts here:
{input}"""

prompt = ChatPromptTemplate.from_messages([("system", template), ("human", "{input}")])

model = ChatOllama()


def _sanitize_output(text: str):
    code = text.split('```')
    print(code)
    return code[1]

chain = prompt | model | StrOutputParser() | _sanitize_output | PythonREPL().run

def autosolve(question):
    results = []
    result_counter = Counter()
    for i in range(100):
        result = chain.invoke({"input": question}).strip()

        print("most recent:", result)

        result_counter.update(result.strip())
        print("most common solution: ", result_counter.most_common(1)[0][0])

def read_problem(day_nr, part_nr):
    return open("data/day%d/part%d.txt"%(day_nr, part_nr)).read()

def main():
    problem = read_problem(1,1)
    autosolve(problem)


if __name__=="__main__":
    main()
