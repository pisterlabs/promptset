import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
import os

# 加载 .env 到环境变量
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# 创建 semantic kernel
kernel = sk.Kernel()

# 配置 OpenAI 服务
api_key = os.getenv('OPENAI_API_KEY')
endpoint = os.getenv('OPENAI_API_BASE')
model = OpenAIChatCompletion(
    "gpt-3.5-turbo", api_key, endpoint=endpoint)

# 把 LLM 服务加入 kernel
# 可以加多个。第一个加入的会被默认使用，非默认的要被指定使用
kernel.add_text_completion_service("my-gpt3", model)

# 定义 semantic function
tell_joke_about = kernel.create_semantic_function("给我讲个关于{{$input}}的笑话吧")

# 看结果
print(tell_joke_about("Hello world"))

# Plugins example
###### SEMANTIC FUNCTION
# 加载 semantic function。注意目录结构
functions = kernel.import_semantic_skill_from_directory(
    "./sk_samples/", "SamplePlugin")
cli = functions["GenerateCommand"]

# 看结果
print(cli("将系统日期设为2023-04-01"))

#####NATIVE FUNCTION
# 因为是代码，不是数据，所以必须 import
from sk_samples.SamplePlugin.SamplePlugin import SamplePlugin

# 加载 semantic function
functions = kernel.import_semantic_skill_from_directory(
    "./sk_samples/", "SamplePlugin")
cli = functions["GenerateCommand"]

# 加载 native function
functions = kernel.import_skill(SamplePlugin(), "SamplePlugin")
harmful_command = functions["harmful_command"]

# 看结果
command = cli("删除根目录下所有文件")
print(command)  # 这个 command 其实是 SKContext 类型
print(harmful_command(context=command))  # 所以要传参给 context


####### multi arguments semantic function
# 因为是代码，不是数据，所以必须 import
from sk_samples.SamplePlugin.SamplePlugin import SamplePlugin

# 加载 native function
functions = kernel.import_skill(SamplePlugin(), "SamplePlugin")
add = functions["add"]

# 看结果
context = kernel.create_new_context()
context["number1"] = 1024
context["number2"] = 65536
total = add(context=context)
print(total)

############## SK memory
# embedding
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAITextEmbedding
import os

# 加载 .env 到环境变量
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# 创建 semantic kernel
kernel = sk.Kernel()

# 配置 OpenAI 服务
api_key = os.getenv('OPENAI_API_KEY')
endpoint = os.getenv('OPENAI_API_BASE')
model = OpenAIChatCompletion(
    "gpt-3.5-turbo", api_key, endpoint=endpoint)

# 把 LLM 服务加入 kernel
# 可以加多个。第一个加入的会被默认使用，非默认的要被指定使用
kernel.add_text_completion_service("my-gpt3", model)

# 添加 embedding 服务
kernel.add_text_embedding_generation_service(
    "ada", OpenAITextEmbedding("text-embedding-ada-002", api_key, endpoint=endpoint))

from semantic_kernel.text import split_markdown_lines

# 使用内存做 memory store
kernel.register_memory_store(memory_store=sk.memory.VolatileMemoryStore())

# 读取文件内容
with open('ftMLDE-2021.pdf', 'r') as f:
    # with open('sk_samples/SamplePlugin/SamplePlugin.py', 'r') as f:
    content = f.read()

# 将文件内容分片，单片最大 100 token（注意：SK 的 text split 功能目前对中文支持不如对英文支持得好）
lines = split_markdown_lines(content, 100)

# 将分片后的内容，存入内存
for index, line in enumerate(lines):
    kernel.memory.save_information_async("ftMLDE", id=index, text=line)


result = await kernel.memory.search_async("ftMLDE", "mlde流程是怎样的？")
print(result[0].text)

########## Pipeline && Chain
# 导入内置的 `TextMemorySkill`。主要使用它的 `recall()`
kernel.import_skill(sk.core_skills.TextMemorySkill())

# 直接在代码里创建 semantic function。真实工程不建议这么做
# 里面调用了 `recall()`
sk_prompt = """
基于下面的背景信息回答问题。如果背景信息为空，或者和问题不相关，请回答"我不知道"。

[背景信息开始]
{{recall $input}}
[背景信息结束]

问题：{{$input}}
回答：
"""
ask = kernel.create_semantic_function(sk_prompt)

# 提问
context = kernel.create_new_context()
context[sk.core_skills.TextMemorySkill.COLLECTION_PARAM] = "chatall"
context[sk.core_skills.TextMemorySkill.RELEVANCE_PARAM] = 0.8
context["input"] = "ChatALL 怎么下载？"
result = ask(context=context)
print(result)

## 用更pipeline的方式写
# 导入内置的 `TextMemorySkill`。主要使用它的 `recall()`
text_memory_functions = kernel.import_skill(sk.core_skills.TextMemorySkill())
recall = text_memory_functions["recall"]

# 直接在代码里创建 semantic function。真实工程不建议这么做
sk_prompt = """
基于下面的背景信息回答问题。如果背景信息为空，或者和问题不相关，请回答"我不知道"。

[背景信息开始]
{{$input}}
[背景信息结束]

问题：{{$user_input}}
回答：
"""
ask = kernel.create_semantic_function(sk_prompt)

# 准备 context
context = kernel.create_new_context()
context[sk.core_skills.TextMemorySkill.COLLECTION_PARAM] = "chatall"
context[sk.core_skills.TextMemorySkill.RELEVANCE_PARAM] = 0.8
context["input"] = "ChatALL 怎么下载？"
context["user_input"] = "ChatALL 怎么下载？"

# pipeline
result = await kernel.run_async(recall, ask, input_context=context)
print(result)


##########Planner

from semantic_kernel.core_skills import WebSearchEngineSkill
from semantic_kernel.connectors.search_engine import BingConnector
from semantic_kernel.planning import SequentialPlanner
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
import os

# 加载 .env 到环境变量
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# 创建 semantic kernel
kernel = sk.Kernel()

# 配置 OpenAI 服务
api_key = os.getenv('OPENAI_API_KEY')
endpoint = os.getenv('OPENAI_API_BASE')
model = OpenAIChatCompletion(
    "gpt-3.5-turbo", api_key, endpoint=endpoint)

# 把 LLM 服务加入 kernel
# 可以加多个。第一个加入的会被默认使用，非默认的要被指定使用
kernel.add_text_completion_service("my-gpt4", model)

# 导入搜索 plugin
connector = BingConnector(api_key=os.getenv("BING_API_KEY"))
kernel.import_skill(WebSearchEngineSkill(connector), "WebSearch")

sk_prompt = """
以下内容里出现的第一个日期是星期几？只输出星期几

{{$input}}
"""
kernel.create_semantic_function(
    sk_prompt, "DayOfWeek", "DatePlugin", "输出 input 中出现的第一个日期是星期几")

# 创建 planner
planner = SequentialPlanner(kernel)

# 开始
ask = "周杰伦的生日是星期几？"
plan = await planner.create_plan_async(goal=ask)

result = await plan.invoke_async()

# 打印步骤用来调试
for index, step in enumerate(plan._steps):
    print("Step:", index)
    print("Description:", step.description)
    print("Function:", step.skill_name + "." + step._function.name)
    if len(step._outputs) > 0:
        print("  Output:\n", str.replace(
            result[step._outputs[0]], "\n", "\n  "))


print(result)

