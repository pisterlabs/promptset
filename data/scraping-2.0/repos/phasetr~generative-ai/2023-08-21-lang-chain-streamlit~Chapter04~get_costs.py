from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
llm = OpenAI(model_name="gpt-3.5-turbo")
with get_openai_callback() as cb:
    result = llm("Tell me a joke")
    print(cb)
