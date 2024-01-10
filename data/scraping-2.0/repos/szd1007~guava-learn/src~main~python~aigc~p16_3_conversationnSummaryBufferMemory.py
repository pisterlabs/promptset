from langchain.memory import ConversationSummaryBufferMemory
import openai, os
from langchain.prompts import PromptTemplate
from langchain.llms import  OpenAI
from langchain.chains import LLMChain
from  langchain.chains import ConversationChain
from langchain.chains import LLMRequestsChain
openai.api_key = os.environ.get("OPENAI_API_KEY")
import re
import json
from langchain.chains import TransformChain, SequentialChain

SUMMARIZER_TEMPLATE = """请将以下内容逐步概括所提供的对话内容，并将新的概括添加到之前的概括中，形成新的概括。
EXAMPLE
Current summary:
Human询问AI对人工智能的看法。AI认为人工智能是一种积极的力量。

New lines of conversation:
Human：为什么你认为人工智能是一种积极的力量？
AI：因为人工智能将帮助人类发挥他们的潜能。

New summary:
Human询问AI对人工智能的看法。AI认为人工智能是一种积极的力量，因为它将帮助人类发挥他们的潜能。
END OF EXAMPLE

Current summary:
{summary}

New lines of conversation:
{new_lines}

New summary:"""

SUMMARY_PROMPT = PromptTemplate(
    input_variables=["summary", "new_lines"], template=SUMMARIZER_TEMPLATE
)

memory = ConversationSummaryBufferMemory(llm=OpenAI(), prompt=SUMMARY_PROMPT, max_token_limit=256)

CHEF_TEMPLATE = """你是一个中国厨师，用中文回答做菜的问题。你的回答需要满足以下要求:
1. 你的回答必须是中文。
2. 对于做菜步骤的回答尽量详细一些。

{history}
Human: {input}
AI:"""

CHEF_PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=CHEF_TEMPLATE
)

conversation_with_summary = ConversationChain(
    llm=OpenAI(model_name="text-davinci-003", max_tokens=2048, temperature=0.5),
    prompt=CHEF_PROMPT,
    memory=memory,
    verbose=True
)
answer = conversation_with_summary.predict(input="你是谁？")

print(answer)
print(conversation_with_summary.predict(input="鱼香肉丝怎么做？"))
print(conversation_with_summary.predict(input="可以放辣椒么？"))
print(conversation_with_summary.predict(input="我问你的第一个问题是啥"))