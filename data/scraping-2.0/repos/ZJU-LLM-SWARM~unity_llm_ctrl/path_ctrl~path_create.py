# -*- coding:utf-8 -*-
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import os
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
import json
from langchain.chains import LLMChain
import re

#路径规划
import numpy as np
openai_api_key = os.environ["OPENAI_API_KEY"]
os.environ['SERPAPI_API_KEY'] = "ecba56657bab25863cd52e445cd4f65214afe50b1507f2a10e423293877f0516"

chat_model = ChatOpenAI(temperature=0, model_name='gpt-4', openai_api_key=openai_api_key)

def path_evaluation(instruction):
    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template("你是一个python测试专家，需要完成对已经生成的python代码的检查和测试，你需要检查该代码的路径规划是否从无人机当前位置出发，是否连贯、是否处于地图范围内，是否合理且保证该路径是能够覆盖全图的最短路径,修正路径规划超出地图边界的情况，最终输出你修改后的代码。该代码的任务为实现无人机对地图的巡视，最终将路径规划结果保存为文件。代码中的起点、地图边界都是正确的，无需更改其中python代码如下： {value}"),
            SystemMessagePromptTemplate.from_template("无人机的扫描区域为半径为100m的圆，无人机在任务过程中，只需要使用numpy数组来表示无人机的路径点，如np.array([[x1,y1,z1],[x2,y2,z2]])，数组长度应当视情况而定，但是数组中的每个元素都应当是float类型，且每步规划的步长应当不小于100m") 
        ],
        input_variables=["value"],
    )

    fruit_query = prompt.format_prompt(value=instruction)
    print (fruit_query.messages[0].content)

    fruit_output = chat_model(fruit_query.to_messages())
    return fruit_output.content

def generate_task(information):
    # # 加载一些要使用的工具
    llm = OpenAI(model_name="gpt-4", openai_api_key=openai_api_key)

    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    # 初始化 Agents
    agent = initialize_agent(tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    environment = "是一个python编程专家，你要用python编写代码来完成无人机的路径规划以完成完整的方型地图快速扫描，最终将路径规划结果保存为txt文件，且每个数组元素之间无需添加符号。在路径规划时，需要注意起点位置和地图边界，路径规划不能超出地图边界。本任务需要最终输出.py文件,"
    input = [environment, information, "无人机的探测范围为半径为100m，无人机在任务过程中高度需保持在25m，只需要使用numpy数组来表示无人机的路径点，其中x,z代表水平方向，y代表竖直方向，即高度。数组长度应当视情况而定，但是数组中的每个元素都应当是float类型"]
    # 测试一下！
    result= agent.run(input)
    return result

def generate_map():
    # # 加载一些要使用的工具
    llm = OpenAI(model_name="gpt-4", openai_api_key=openai_api_key)

    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    # 初始化 Agents
    agent = initialize_agent(tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    input = "两架无人机合作实现对地图的探测，地图长宽为1000*1000,地图的4角边缘分别为[-500,0,-500],[-500,0,500],[500,0,500],[500,0,-500]，坐标系原地位于地图中央，无人机1位置坐标为[-500,25,-500],无人机2位置坐标为[500,25,500],你需要将地图分割为两部分(应当尽量保持矩形)，分别由两架无人机进行探测，通过生成python代码将两家无人机需要探索的地图信息和自身位置信息分别保存在map1.txt map2.txt中" 
    # 测试一下！
    result= agent.run(input)
    return result

def gerate_path_style():
    # # 加载一些要使用的工具
    llm = OpenAI(model_name="gpt-4", openai_api_key=openai_api_key)

    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    # 初始化 Agents
    agent = initialize_agent(tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    input = "想要实现无人机对正方形区域的快速扫描侦查，蛇形搜索是否合适？亦或者应当采用其他路径规划方式？当前扫描任务十分简单，应当选取那种方式最合适？一定要选择一种方式作为结果，说明理由" 
    # 测试一下！
    result= agent.run(input)
    return result

with open('./situation.txt', 'r') as file:
    situation = file.read()
with open('./drone1_map.txt', 'r') as file:
    drone1_map = file.read()
with open('./drone2_map.txt', 'r') as file:
    drone2_map = file.read()

# generate_map()
gerate_path_style_result = gerate_path_style()
# gerate_path_style_result = "蛇形规划"
# information = "路径规划文件名应当保存为003.txt。使用的规划方式：" + gerate_path_style_result + "规划路线应当从当前位置开始，且步长设定为100。地图信息为" + str(drone1_map)
# path_result = generate_task(information=information)
# eva_result = path_evaluation(path_result)
# information2 = "路径规划文件名应当保存为004.txt。使用的规划方式：" + gerate_path_style_result + "规划路线应当从当前位置开始，且步长设定为200。地图边界点和无人机位置为：" + str(drone2_map) +"。无人机位于地图右下角"
# path_result2 = generate_task(information=information2)
# eva_result2 = path_evaluation(path_result2)
