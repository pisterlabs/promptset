#Prompt templates 提示模版
from langchain import PromptTemplate,FewShotPromptTemplate
from langchain.llms import OpenAI
import os


#示例一：普通提示（Normal Prompting）
template = """
你喜欢{city}这座城市嘛?
"""
os.environ["OPENAI_API_KEY"] = "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"
prompt = PromptTemplate.from_template(template)
str = prompt.format(city="武汉")
print(str)
print("--------------------------------------------------")
#示例二：零样本提示（Zero-Shot Prompting）
"""这个例子很简单，是一个零样本提示，LangChain的PromptTemplate组件做了文本替换，把lastname替换成了用户输入，模型接收提示，返回了结果，。"""
template = "我的邻居姓{lastname}，他生了个儿子，给他儿子起个名字"

prompt = PromptTemplate(
    input_variables=["lastname"],
    template=template,
)

prompt_text = prompt.format(lastname="王")
# result: 我的邻居姓王，他生了个儿子，给他儿子起个名字

# 调用OpenAI
llm = OpenAI(temperature=0.9)
print(llm(prompt_text))
print("--------------------------------------------------")
#示例三：小样本提示（Few-Shot Prompting）
"""我们再看一个例子，需求是根据用户输入，让模型返回对应的反义词，我们要通过示例来告诉模型什么是反义词，这是一个小样本提示，我们就用到了LangChain的FewShotPromptTemplate 组件。"""
examples = [
    {"word": "开心", "antonym": "难过"},
    {"word": "高", "antonym": "矮"},
]
example_template="""
单词：{word}
反义词：{antonym}
"""
example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template=example_template,
)
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="给出每个单词的反义词",
    suffix="单词: {input}\n反义词:",
    input_variables=["input"],
    example_separator="\n",
)

prompt_text = few_shot_prompt.format(input="粗")
print(prompt_text)

# 调用OpenAI
llm = OpenAI(temperature=0.9)
print(llm(prompt_text))





