
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)

# AI会去总结一下前面几轮的对话
memory = ConversationSummaryMemory(llm=OpenAI())

prompt_template = """你是一个中国厨师，用中文回答做菜的问题。你的回答需要满足以下要求:
1. 你的回答必须是中文
2. 回答限制在100个字以内

{history}
Human: {input}
AI:"""
prompt = PromptTemplate(
    input_variables=["history", "input"], template=prompt_template
)

# 在我们打开了 ConversationChain 的 Verbose 模式，然后再次询问 AI 第二个问题的时候，你可以看到，在 Verbose 的信息里面，没有历史聊天记录，而是多了一段对之前聊天内容的英文小结。
conversation_with_summary = ConversationChain(
    llm=llm, 
    memory=memory,
    prompt=prompt,
    verbose=True
)

answer = conversation_with_summary.predict(input="你好")
print(answer)
answer = conversation_with_summary.predict(input="鱼香肉丝怎么做？")
print(answer)

# 调用 memory 的 load_memory_variables 方法，可以看到记录下来的 history 是一小段关于对话的英文小结
memory.load_memory_variables({})

# 我们进一步通过 conversation_with_summary 去和 AI 对话，就会看到英文的小结内容会随着对话内容不断变化。
answer = conversation_with_summary.predict(input="那宫保鸡丁呢？")
print(answer)

# 虽然 SummaryMemory 可以支持更长的对话轮数，但是它也有一个缺点，就是即使是最近几轮的对话，记录的也不是精确的内容。当你问“上一轮我问的问题是什么？”的时候，它其实没法给出准确的回答。