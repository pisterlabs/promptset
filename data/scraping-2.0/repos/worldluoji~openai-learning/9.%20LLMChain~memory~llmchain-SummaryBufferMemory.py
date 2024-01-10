
from langchain import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.llms import OpenAI

# 把 Langchain 原来默认的对 Memory 进行小结的提示语模版从英文改成中文的了，不过这个翻译工作我也是让 ChatGPT 帮我做的。如果你想了解原始的英文提示语是什么样的，可以去看一下它源码里面的 _DEFAULT_SUMMARIZER_TEMPLATE : https://github.com/hwchase17/langchain/blob/master/langchain/memory/prompt.py
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

# 打开了 Verbose 模式，所以你能看到实际 AI 记录的整个对话历史是怎么样的。当我们连续多问 AI 几句话，你就会看到，随着对话轮数的增加，Token 数量超过了前面的 max_token_limit 。于是 SummaryBufferMemory 就会触发，对前面的对话进行小结，也就会出现一个 System 的信息部分，里面是聊天历史的小结，而后面完整记录的实际对话轮数就变少了。
conversation_with_summary = ConversationChain(
    llm=OpenAI(model_name="text-davinci-003", stop="\n\n", max_tokens=2048, temperature=0.5), 
    prompt=CHEF_PROMPT,
    memory=memory,
    verbose=True
)
answer = conversation_with_summary.predict(input="你是谁？")
print(answer)


answer = conversation_with_summary.predict(input="请问鱼香肉丝怎么做？")
print(answer)

# 再问蚝油牛肉，前面的对话就被小结到 System 下面去了
answer = conversation_with_summary.predict(input="那蚝油牛肉呢？")
print(answer)