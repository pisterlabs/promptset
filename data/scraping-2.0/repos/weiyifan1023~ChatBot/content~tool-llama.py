import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["OPENAI_API_KEY"] = "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"
os.environ['HTTP_PROXY'] = "http://cipzhao:cipzhao@210.75.240.136:10800"
os.environ['HTTPS_PROXY'] = "http://cipzhao:cipzhao@210.75.240.136:10800"
import sys

sys.path.append("..")
import re
import argparse
import openai
from time import sleep
from langchain.agents.tools import Tool
from langchain.agents import AgentType, initialize_agent
from langchain.tools.python.tool import PythonREPLTool
from models.llama_llm import Llama
from models.chatglm_llm import ChatGLM  # custom LLM
from configs.model_config import *  # 配置
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

parser = argparse.ArgumentParser()
# data number
parser.add_argument("--start", default=14, type=int)
parser.add_argument("--end", default=16, type=int)
# data type
parser.add_argument("--update", default=1, type=int)  # 对照实验记得改这里！！！
parser.add_argument("--counterfactual", default=0, type=int)
parser.add_argument("--disorder", default=0, type=int)
parser.add_argument("--all_update", default=None, type=int)
# select base LLM
parser.add_argument("--llm_model", default="gpt-3.5-turbo", type=str)  # 选择 base LLM
args = parser.parse_args()

turbo_llm = ChatOpenAI(
    temperature=0,
    model_name='gpt-3.5-turbo',
)



# time比较工具
class PythonCompareTime(BaseTool):
    name = "Python Comparison Time Tool"
    description = (
        "Useful for when you need to answer a question about time, "
        # "Useful when you want to compare two sets of time scopes."
        "The input of the tool should be two sets of time intervals, each containing a start and an end time, "
        "like: ['2003-10','2006'], ['2004-9', '2005'] "
        "representing the time interval in the question being asked and "
        "the time interval of when the event mentioned in the question occurred in the context, respectively."
    )
    print("打印tool description: ", description)

    return_direct = False  # 不直接返回结果

    def _run(self, input: str) -> str:
        """extract time scopes from context and question, and compare two sets of time scopes"""
        flag = False
        output = ""
        print("\nMy Tool Input: ", input)
        print("Input type: ", type(input))
        try:
            input = eval("[" + input + "]")
            query_time = input[0]
            context_time = input[1]
            print("query time: ", input[0])
            print("context time: ", input[1])
            if query_time[0] >= context_time[0] and query_time[1] <= context_time[1]:
                flag = True
                equal_question = ""
            print("成功执行Python time comparison tool !")

        except Exception as e:
            print('错误类型是', e.__class__.__name__)
            print('异常抛出错误明细是', e)
            flag = False

        if flag == True:
            # output = "The time scope mentioned in the question belongs to the time interval of real events in the context, " \
            #          "so it can be answered using information from the context."
            output = f"['{context_time[0]}', '{context_time[1]}']"
        else:
            output = "The answer is: unanswerable"
            self.return_direct = True

        return output

    async def _arun(self, input: str) -> str:
        raise NotImplementedError("暂时不支持异步")


# Equivalent transform 工具
class QueryTransform(BaseTool):
    name = "Query Transform Tool"
    description = (
        "Useful for when you need to use an equivalent question to arrive at an answer, "
        "The input of the tool should be a original question and a time scope, "
        "Like: ['Dean Holdsworth played for which team from 1986 to 1989?', ['2003-10','2006']], "
        "Return an equivalent question, and then continue to answer based on the equivalent question."
    )
    print("打印tool description: ", description)

    return_direct = False  # 直接返回结果

    def _run(self, input: str) -> str:
        """Transform the original question into an equivalent new question, and answer based on the new question."""
        print("\nMy Tool Input: ", input)
        print("Input type: ", type(input))
        try:
            input = eval(input)
            question = input[0]
            time_list = input[1]
            print("query : ", input[0])
            print("time scope: ", input[1])
            time_match = re.findall(r"([0-9]{4})(-\d{2})?", question)
            print("time_ match", time_match)
            query_time_list = ["".join(t) for t in time_match]
            print("query_time_list: ", query_time_list)
            # 将时间点按顺序用列表中的时间替换
            new_question = question.replace(query_time_list[0], time_list[0]).replace(query_time_list[1], time_list[1])
            print("成功执行Python time comparison tool !")
            self.return_direct = True
            return new_question

        except Exception as e:
            print('错误类型是', e.__class__.__name__)
            print('异常抛出错误明细是', e)

        return ""

    async def _arun(self, input: str) -> str:
        raise NotImplementedError("暂时不支持异步")




instruction = "Instruction: Answer the question based on context, with answers derived from substrings in the context or categorized as 'unanswerable':\n"
query = "What was Lothar von Trotha afflicted to in 1895?"
context = "From 1893-10 to 1898-09 Trotha was appointed commander of the colonial forces in German East Africa and was ruthlessly successful in suppressing uprisings there , including the Wahehe Rebellion . While temporarily posted to Imperial China as Brigade Commander of the East Asian Expedition Corps , he was involved in suppressing the Boxer Rebellion . On 3 May 1904 he was appointed Commander in Chief of German South West Africa and was directed to crush the native Herero rebellion . In German South West Africa ."
input_sample = f'Context: {context}\nQuestion: {query}\nAnswer:'

tools = [PythonCompareTime(), QueryTransform()]

llm = ChatGLM()
# llm.load_model(model_name_or_path=llm_model_dict[LLM_MODEL], llm_device=LLM_DEVICE)
llm = turbo_llm
llm = Llama()
llm.load_model(model_name_or_path="/home/weiyifan/LLMs/langchain_ChatGLM/checkpoints/llama-13b-hf")
agent_chain = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
print("输入: ", input_sample)
new_query = agent_chain.run(input_sample)
print("original query: ", query)
print("equivalent new_query: ", new_query)

input_sample = f'Context: {context}\nQuestion: {new_query}\nAnswer:'
# feed to LLMs and  Generate

prompt = instruction + input_sample
response = "output_none"
got_result = False
while not got_result:
    try:
        # ChatCompletion
        if args.llm_model == "gpt-3.5-turbo":
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0.0,  # 0.0 to 2.0 (default 1.0)
                top_p=1,  # 0.0 to 1.0 (default 1.0) (not used if temperature is set)
                n=1,  # number (default 1) How many chat completion choices to generate for each input message.
                stream=False,  # boolean (default False)
                stop=["\n\n"],  # string or array (default None)
                # 我们使用stop字段来控制生成的文本长度和格式。我们指定了两个停止标记，即换行符和"Here are some recommendations:"，
                # 当模型生成文本中出现这些标记时，它将停止生成并返回生成的文本。这样，我们可以确保返回的文本不会太长，并按预期格式进行格式化。
                max_tokens=25,  # inf (default 4096-prompt_token)
                presence_penalty=0,  # -2.0 to 2.0 (default 0)
                frequency_penalty=1,  # -2.0 to 2.0 (default 0)
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": input_sample},
                ]
            )
            response = completion.choices[0].message.content
            # Completion
        else:
            completion = openai.Completion.create(
                engine=args.llm_model,
                prompt=prompt,
                max_tokens=50,
                temperature=0,
                logprobs=1,
                stop=["\n\n"]
            )
            response = completion['choices'][0]['text']
        # api访问失败，循环请求
        got_result = True
    except Exception as e:
        sleep(3)
        print('sleep 5 !  错误类型是', e.__class__.__name__)
        print('错误明细是', e)

print('模型答案:', response)
