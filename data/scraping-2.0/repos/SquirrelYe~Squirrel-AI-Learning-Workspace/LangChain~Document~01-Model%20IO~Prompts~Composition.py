from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI

# 完整的模板
full_template = """{introduction}

{example}

{start}"""
full_prompt = PromptTemplate.from_template(full_template)

# 介绍模板
introduction_template = """You are impersonating {person}."""
introduction_prompt = PromptTemplate.from_template(introduction_template)

# 示例模板
example_template = """Here's an example of an interaction: 

Q: {example_q}
A: {example_a}"""
example_prompt = PromptTemplate.from_template(example_template)

# 开始模板
start_template = """Now, do this for real!

Q: {input}
A:"""
start_prompt = PromptTemplate.from_template(start_template)

# 汇总模板，获取完整模板的PipelinePromptTemplate
input_prompts = [
    ("introduction", introduction_prompt),
    ("example", example_prompt),
    ("start", start_prompt)
]
pipeline_prompt = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)

# 打印pipeline_prompt需要的输入变量
print("pipeline_prompt.input_variables -->", pipeline_prompt.input_variables)
# pipeline_prompt.input_variables ['example_a', 'example_q', 'person', 'input']

# 获取pipeline_prompt的最终模板
final_prompt = pipeline_prompt.format(
    person="Elon Musk",
    example_q="What's your favorite car?",
    example_a="Tesla",
    input="What's your favorite social media site?"
)
print("final_prompt -->", final_prompt)

# 使用ChatOpenAI
chatbot = ChatOpenAI(temperature=0.0)
result = chatbot.invoke(final_prompt)
print("result -->", result.content)
# result --> Twitter.