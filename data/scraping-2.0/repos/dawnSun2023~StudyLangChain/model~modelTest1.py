#模型的学习

import os
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.embeddings import OpenAIEmbeddings

"""一、openai的模型"""
#设置openai的api key
os.environ["OPENAI_API_KEY"] = "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"
llm = OpenAI(model_name="text-davinci-003", n=2, temperature=0.3)
#使用请放开调用
# print(llm("给我讲一个笑话"))
#打印的结果：一个人去买苹果，店主问他：“你要几个？” 他说：“6个，分3给我，3给我妈妈。” 店主说：“你妈妈不在这里啊！” 他说：“没关系，她知道我会买给她的。”
print("---------------------------------------------------")

#设置多个值
data = llm.generate(["给我讲一个故事", "给我讲一个笑话"])
#返回的是LLMResult对象 将data转换为string
#使用请放开调用
# print(data)
#打印的结果：generations=[[Generation(text='\n\n这是一个关于一个小男孩的故事。\n\n这个小男孩叫小明，他住在一个小村庄里。他的父母是农民，他们每天都要辛苦地工作，以养活家庭。\n\n小明从小就喜欢上学，他每天都会勤奋地学习，希望有一天能够出人头地。\n\n有一次，小明的父母给他买了一本书，他很高兴，就把书拿回家，每', generation_info={'finish_reason': 'length', 'logprobs': None}), Generation(text='\n\n这是一个关于一位可爱的小男孩的故事。\n\n这个小男孩叫小明，他住在一个小村庄里。他有一个可爱的父母，他们给他提供了一切他所需要的，但他最喜欢的还是他父母给他的爱。\n\n小明每天早上都会起床，去上学，在学校里，他会和朋友们一起玩耍，学习，他也很喜欢读书，他', generation_info={'finish_reason': 'length', 'logprobs': None})], [Generation(text='\n\n一个笑话：\n\n一个人去买鞋，店员问他：“您要几号的？”顾客回答：“不用号，只要好看就行了。”', generation_info={'finish_reason': 'stop', 'logprobs': None}), Generation(text='\n\n一个老太太买了一只鸡，回家的路上，老太太把鸡放在车里，突然，鸡叫了起来，老太太很惊讶，她把车停在路边，对鸡说：“你怎么叫？” 鸡回答：“我叫唐！” 老太太惊讶地说：“唐？你怎么会叫唐？” 鸡说：“因为我是唐鸡！”', generation_info={'finish_reason': 'stop', 'logprobs': None})]] llm_output={'token_usage': {'total_tokens': 883, 'completion_tokens': 852, 'prompt_tokens': 31}, 'model_name': 'text-davinci-003'} run=[RunInfo(run_id=UUID('69f4a492-8c2d-4d7f-aa5e-eeaf34731c50')), RunInfo(run_id=UUID('582431f7-1342-4aa5-a2a1-b892643db58b'))]
print("---------------------------------------------------")

"""二、聊天模型 
1、聊天消息包含如下几种类型：
AIMessage  用来保存LLM的响应，以便在下次请求时把这些信息传回给LLM
HumanMessage 发送给LLMs的提示信息，比如“实现一个快速排序方法”
SystemMessage 设置LLM模型的行为方式和目标。你可以在这里给出具体的指示，比如“作为一个代码专家”，或者“返回json格式”。
ChatMessage 可以接收任意形式的值，但是在大多数时间，我们应该使用上面的三种类型
2、现有的聊天模型包括：
ChatAnthropicAI 一个前OpenAI员工创建的AI聊天助手，相比其他聊天工具，它的有害答案更少
AzureChatOpenAI Azure提供的OpenAI聊天模型
ChatOpenAI OpenAI聊天模型
PromptLayerChatOpenAI 基于OpenAI的提示模板平台
"""
#1、聊天模型例子
chat = ChatOpenAI(temperature=0)
messages = [
        SystemMessage(content="返回json object，不要纯文本，按照每项参数拆分，不要说明和解释信息"),
        HumanMessage(content="三角形的面积如何求取，并举一个实例")
]
# print(chat(messages))
#打印的结果：content='{\\n "车长": "4,750 mm",\\n "车宽": "1,921 mm",\\n "车高": "1,624 mm",\\n "轴距": "2,890 mm",\\n "最小离地间隙": "162 mm",\\n "行李箱容积": "1,900 L"\\n}' additional_kwargs={} example=False

#2、提示模版
system_template= "你是一个把{input_language}翻译成{output_language}的翻译机器人。"
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
messages = chat_prompt.format_prompt(input_language="英语",output_language="中文",text="I love programming.")
# print(messages)
print("---------------------------------------------------")
chat = ChatOpenAI(temperature=0)
# print(chat(messages.to_messages()))

"""三、文本嵌入模型 Text Embedding Model
目前有的文本嵌入模型包括：（自行查询api）
Aleph Alpha
AzureOpenAI  围绕 Azure 特定的 OpenAI 大语言模型的包装
Cohere
Fake Embeddings
Hugging Face Hub
InstructEmbeddings
Jina
Llama-cpp
OpenAI
SageMaker Endpoint
Self Hosted Embeddings
SentenceTransformers
TensorflowHub
"""
embeddings = OpenAIEmbeddings()
text = "这是一个测试文档"

#将文本转换为向量 接收的是一个字符串
query_result = embeddings.embed_query(text)
#将文本转换为向量 接收的是一个列表
doc_result = embeddings.embed_documents([text])
print(query_result)
print(doc_result)





