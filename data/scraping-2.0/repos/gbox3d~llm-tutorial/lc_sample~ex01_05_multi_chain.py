#%%
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain.callbacks import StreamingStdOutCallbackHandler

import time
import os
from dotenv import load_dotenv
load_dotenv('../.env')

chat = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler(),
    ]
)

print("chat model ready")

#%%
chef_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 요리사입니다. "),
        ("human", "나는 {cuisine} 요리를 만들고 싶어요")
    ]
)
chef_chain = chef_prompt | chat

# answer = chef_chain.invoke({
#     "cuisine": "한식"
# })

# %%

veg_chef_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", """
            당신은 전통 요리를 채식주의자들이 먹을 수 있도록 변형하는 전문 채식 요리사입니다. 
            전통적인 재료를 대체할 수 있는 채식 재료를 찾아 그 준비 방법을 설명해 주세요. 
            그러나 레시피를 근본적으로 변경하지는 않습니다.
            만약 대체할 수 없는 재료가 있다면 할수없다고 말해주세요.
            """
        ),
        ("human", "{recipe}"),
    ]
)

veg_chef_chain = veg_chef_prompt | chat

#%%
final_chain = {"recipe" : chef_chain} | veg_chef_chain

finally_answer = final_chain.invoke({"cuisine": "한식 불고기"})
#%%
print(finally_answer)

# %%
