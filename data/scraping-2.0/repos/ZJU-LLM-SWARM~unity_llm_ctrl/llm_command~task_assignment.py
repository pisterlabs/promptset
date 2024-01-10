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

openai_api_key = os.environ["OPENAI_API_KEY"]
os.environ['SERPAPI_API_KEY'] = "ecba56657bab25863cd52e445cd4f65214afe50b1507f2a10e423293877f0516"

chat_model = ChatOpenAI(temperature=0, model_name='gpt-4', openai_api_key=openai_api_key)

def generate_sub_task(instruction):

    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template("你是一个python编程专家，根据用户内容,将任务转换成可执行的python文件，且python文件中应当包含以下函数：(1)执行函数：do(task_id)，该函数可以实现在传入不同的task_id时执行不同的任务内容；(2)检测函数standard(task_id):该函数可以实现对当前子任务是否完成的判断，完成时返回1，未完成时返回0.(3)get_state(enironment)该函数可以实现环境信息的获取，并将结果返回到无人系统当中(4)其他函数：当需要完成特定功能时，可以创建其他函数 {value}"),
            SystemMessagePromptTemplate.from_template("在任务过程中，只需要使用numpy数组来表示无人机和无人车的轨迹规划即可实现对无人车和无人机的控制，如无人车的轨迹规划：np.array([[-100,-100],[-100,0],[0,0]])，无人机的轨迹规划：np.array([[-100,25,-100],[-100,25,0],[0,25,0]])。无人车和无人机观测目标的函数已经集成在无人车和无人机自身，可以在situation.txt中读取。所有的环境、无人机和无人车、目标状态都保存在./situation.txt文件中")
            #                                             car1:position:[-310.715, -0.03285286, -320.2984]rotation:[0.002093262, 48.7872, -0.0003284954]speed:4.950579target_found_distance[0.0, 444.7502] \\
            #                                             car2: \\
            #                                             position:[310.3896, -0.03275179, 320.486]rotation:[0.004431482, 228.8786, -0.0003165574]speed:6.221848target_found_distance[0.0, 447.685] \\
            #                                             drone1: \\
            #                                             position[0.1514675, 25.0, 0.1723792]rotation:[1.064369e-05, 56.50008, 2.508069e-06]target_found_distance[1.0, 4.235265] \\
            #                                             drone2: \\
            #                                             position[-0.1094773, 25.0, -0.07250695]rotation:[1.192111e-05, 213.5, 5.707033e-06]target_found_distance[1.0, 4.1171] \\
            #                                             target:position[2.595578, 0.0, -4.214604]； "                                
                                                        
        ],
        input_variables=["value"],
    )

    fruit_query = prompt.format_prompt(value=instruction)
    print (fruit_query.messages[0].content)

    fruit_output = chat_model(fruit_query.to_messages())
    return fruit_output.content

# 找到全部的task文本
def str_to_subtask(str):
    pattern = r'{.*?}'
    matches = re.findall(pattern, str)
    return matches

def task_plan(instruction):
    chat_model = ChatOpenAI(temperature=0, model_name='gpt-4', openai_api_key=openai_api_key)

    response_schemas = [
        ResponseSchema(name="task_id", description="task_id是子任务的唯一标识符，用阿拉伯数字表示顺序"),
        ResponseSchema(name="summary_id", description="子任务的内容简短总结(10字以内)"),
        ResponseSchema(name="task_id_content", description="阐述分割出的子任务内容"),
        ResponseSchema(name="standard_id", description="判断任务是否完成的标准，尽可能量化"),
        ResponseSchema(name="select_task", description="所有任务的之间的逻辑关系，例如：如果任务1达到其完成标准，则执行任务2，否则继续执行任务1或者采取其他动作；表示形式采用如下的python形式：start(); do(task_id);if standard(task_id): do(task_id); else： do(task_id);  其中standard(task_id)为判断任务是否完成的函数，当任务完成时返回1，未完成时返回0，do(task_id)为执行任务的函数，start(),end()分别为结束整个任务的函数，其中需要确保所有任务目标完成后才能调用end()。其中数字为任务的task_id，并将最终的输出转换为python的逻辑代码"),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template("根据用户内容,按照任务内容将用户内容拆分为可单独执行的几个子任务,同时将语义相近的任务进行合并，并精炼提取相应的任务动作，给出子任务完成的标准，即满足该标准时，认为子任务完成，用中文回答。{format_instructions}用户输入: {user_prompt}")
        ],
        input_variables=["user_prompt"],
        partial_variables={"format_instructions": format_instructions},
    )

    user_prompt = instruction
    fruit_query = prompt.format_prompt(user_prompt=user_prompt)
    # print('输入内容：', fruit_query.messages[0].content)

    fruit_output = chat_model(fruit_query.to_messages())
    # print('LLM 输出：', fruit_output.content)
    return fruit_output.content

def logical_reasoning(tasks):
    chat_model = ChatOpenAI(temperature=0, model_name='gpt-4', openai_api_key=openai_api_key)

    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template("你是一个python编程专家，根据用户所给的包含python命令的任务内容，汇总成一个python文件，判断当前逻辑是否满足任务要求，是否存在逻辑问题，如果存在逻辑问题则进行修改，并根据任务内容进行注释，用户输入{user_prompt}")
        ],
        input_variables=["user_prompt"],
    )

    user_prompt = tasks
    fruit_query = prompt.format_prompt(user_prompt=user_prompt)
    # print('输入内容：', fruit_query.messages[0].content)

    fruit_output = chat_model(fruit_query.to_messages())
    # print('LLM 输出：', fruit_output.content)
    return fruit_output


def generate_task(instruction):
    # # 加载一些要使用的工具
    llm = OpenAI(model_name="gpt-4", openai_api_key=openai_api_key)

    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    # 初始化 Agents
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    environment = "当前地图为1000×1000的二维平面地图，其中坐标系零点在地图中心位置；当前可供支配的设备包括两台无人车和两架无人机。 无人车：可以实现细粒度高的侦查和抓捕，但移动速度较慢,可达范围有限，隐蔽性弱，无人车上具备生命探测装置，探测距离为30m。无人机：探索范围大细粒度较低，但移动速度快，隐蔽性强。无人机具有热成像相机，可以在高空探测半径100m内的物体，但精准度差，可能误识别。" 
    input = [environment, instruction,"最好给出任务完成标准和完整的任务逻辑，包括任务之间的触发条件，并用中文回答"]
    # 测试一下！
    result= agent.run(input)

    # print(result_json)
    task_planning_result = task_plan(result)
    print(task_planning_result)
    sub_task_py = generate_sub_task(task_planning_result)
    print(sub_task_py)
    # tasks = str_to_subtask(task_planning_result)
    # print(tasks)
    total = task_planning_result + sub_task_py
    print(logical_reasoning(total).content)

generate_task("如何使用无人机和无人车搜索并围捕移动目标？按照任务进行的先后顺序给出详细步骤")
