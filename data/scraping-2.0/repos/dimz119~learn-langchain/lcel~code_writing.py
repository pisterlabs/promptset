from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
)
from langchain.schema import StrOutputParser
from langchain.utilities import PythonREPL

# Introduce PythonREPL
python_repl = PythonREPL()
print(python_repl.run("print(1+1)"))
"""
Python REPL can execute arbitrary code. Use with caution.
2
"""

template = """Write some python code to solve the user's problem. 

Return only python code in Markdown format, e.g.:

```python
....
```"""
prompt = ChatPromptTemplate.from_messages([("system", template), ("human", "{input}")])

model = ChatOpenAI()

def _sanitize_output(text: str):
    _, after = text.split("```python")
    return after.split("```")[0]

chain = prompt | model | StrOutputParser()

print(chain.invoke({"input": "Write the function to sort the list. Then call the function by pasing [1,4,2]"}))
"""
```python
def sort_list(lst):
    return sorted(lst)

my_list = [1, 4, 2]
sorted_list = sort_list(my_list)
print(sorted_list)
```
"""
repl_chain = chain | _sanitize_output | PythonREPL().run

print(repl_chain.invoke({"input": "Write the function to sort the list. Then call the function by pasing [1,4,2]"}))
"""
Python REPL can execute arbitrary code. Use with caution.
[1, 2, 4]
"""
