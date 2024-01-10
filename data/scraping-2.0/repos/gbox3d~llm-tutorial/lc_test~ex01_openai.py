#%%
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv
import os
# .env 파일 로드
load_dotenv()

print(f'OpenAI API key: {os.getenv("OPENAI_API_KEY")}')

#%%

model = "gpt-3.5-turbo-1106"

if os.getenv("OPENAI_API_KEY") is not None:
    model = os.getenv("OPENAI_API_KEY")

llm = OpenAI()
chat_model = ChatOpenAI(
    model="",
    temperature=0.1
)

#%%
text = "지구의 위성은 무엇이 있을까요?"


_answer = llm.invoke(text)
print(_answer)


#%%
messages = [HumanMessage(content=text)]
_answer = chat_model.invoke(messages)
print(_answer)


# %%
