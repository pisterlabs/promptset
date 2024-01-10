import openai 
from pydantic import BaseModel
from typing import List
import json

from pydantic import BaseModel
from typing import List

class Item(BaseModel):
    id: str
    parent_id: str
    name: str

class ItemList(BaseModel):
    # GPT_thinking: str
    items: List[Item]

class Item_Name_list(BaseModel):
    item_names: List[str]

class Container_moving(BaseModel):
    container_name: str 
    new_parent_name: str

class Human_command(BaseModel):
    command: str



def better_human_input(human_input, model='gpt-4'):
    prompt=f"""
你将按照物品容纳的顺序整理我的语音输入记录。
例如
铅笔放在铅笔盒里，铅笔盒放在书包里，书包放在桌子上，桌子有抽屉，抽屉里还放着书，桌子是书房里的。
应当整理成：
根节点有书房，书房里有桌子，桌子里有书包，桌子里有抽屉，书包里有铅笔盒，铅笔盒里有铅笔，抽屉里有书。

以下是我的输入记录：
{human_input}

物品收纳记录如下：
"""

    answer=openai.ChatCompletion.create(
        model=model,
        messages=[
        {'role':'system',"content":"You are a helpful assistant with IQ=120"},
        {"role": "user", "content": prompt}
        ],
        temperature=0,
    )
    better_input=answer.choices[0].message.content
    return better_input

def structured_input(human_input,
        model1='gpt-4',
        model2="gpt-3.5-turbo-16k",
        # model2="gpt-4",
        current_id=10, 
        known_containers="",
        debug=False
        ):
    better_input=better_human_input(human_input, model=model1)

    if debug:
        print(better_input)

    prompt=prompt=f"""
你是一位专业的仓库保管员，负责记录物品的存放位置。我将提供物品存放记录，你的任务是：

* 将记录整理为CSV文件，用于描述物品的存放位置。CSV文件中，每个节点代表一个容器（如房间、柜子等）或物品，通过树形结构表示物品的存放位置。每个节点有唯一标识符（id）、父节点标识符（parent_id）、名称（name）。
下面是一个CSV文件示例：

id,parent_id,name
1,,卧室
2,1,柜子
3,2,抽屉
4,3,手机

这表示"手机"被存放在"卧室"的"柜子"的"抽屉"中。

现在，你需要将以下记录整理为CSV文件。id=1总是根节点，房屋的父节点是根节点。新的物品或容器应有新的id，当前id已用到{current_id}，之前的序号已经被占用。

已知的容器有：

{known_containers}

请注意，新的物品或容器必须放入已知的容器中。
已知的容器无需重复描述。

新的记录如下：
###
{better_input}
###
"""
    if debug:
        print()
        print(prompt)

    response=openai.ChatCompletion.create(
        model=model2,
        messages=[
        {"role": "user", 
        "content": prompt}],
        functions=[
            {"name":"get_storage_recorder",
            "description": "Getting a record of what a user has stored",
            "parameters":ItemList.schema()
            }
        ],
        function_call={"name":"get_storage_recorder"},
        temperature=0.2,
    )
    output = json.loads(response.choices[0]["message"]["function_call"]["arguments"])
    return output['items']

def better_query(human_query, model='gpt-4'):
    prompt=f"""
我在查询物品，但我不记得物品的精确名称了，请输出3个最可能的物品查询关键词。
比如：
泳镜，可能会被称为潜水镜、面镜、游泳镜，这几个词可能会混用

以下是我的查询：
{human_query}
"""

    answer=openai.ChatCompletion.create(
        model=model,
        messages=[
        {'role':'system',"content":"You are a helpful assistant with IQ=120"},
        {"role": "user", "content": prompt}
        ],
        temperature=0,
    )
    return answer.choices[0].message.content

def query_item(
        human_query,
        ref_path="", 
        model='gpt-3.5-turbo-16k'):
    prompt=f"""
请参考下面的物品收纳记录，回答我的问题：
{human_query}

物品收纳记录如下：
{ref_path}
"""

    answer=openai.ChatCompletion.create(
        model=model,
        messages=[
        {'role':'system',"content":"You are a helpful assistant with IQ=120"},
        {"role": "user", "content": prompt}
        ],
        temperature=0,
    )
    return answer.choices[0].message.content

def structured_container_mover(
        human_command,
        model='gpt-3.5-turbo'):
    
    response=openai.ChatCompletion.create(
        model=model,
        messages=[
        {"role": "user", 
        "content": human_command}],
        functions=[
            {"name":"move_from_container_to",
            "description": "Get the name of the container and the destination of the container",
            "parameters":Container_moving.schema()
            }
        ],
        function_call={"name":"move_from_container_to"},
        temperature=0,
    )
    output = json.loads(response.choices[0]["message"]["function_call"]["arguments"])
    return output

def structured_takeout_items(
        human_input,
        model='gpt-3.5-turbo'):
    # 注意，这里缺乏对提取物品的名称检查，可能不存在。
    response=openai.ChatCompletion.create(
        model=model,
        messages=[
        {"role": "user", 
        "content": human_input}],
        functions=[
            {"name":"get_takeout_items_name",
            "description": "Get a list of the names of the removed items",
            "parameters":Item_Name_list.schema()
            }
        ],
        function_call={"name":"get_takeout_items_name"},
        temperature=0,
    )
    output = json.loads(response.choices[0]["message"]["function_call"]["arguments"])
    return output["item_names"]

class Command(BaseModel):
    name: str
    Human_command: str

class CommandList(BaseModel):
    commands: List[Command]

def interpret_human_command(human_command, model='gpt-4'):
    prompt="""
    请解读用户的指令，用户可以进行4种操作：
    1. add_item_from_human_input
    2. query_item_from_human_query
    3. move_container_by_human_command
    4. takeout_items_by_human_command
    比如：
    用户放入物品A，将容器B移动到容器C，然后取出物品E，又放入物品F, 询问物品G的位置。
    则应当解析为：
    [
        {"name":"add_item_from_human_input"
            "Human_command": "放入物品A"},
        {"name":"move_container_by_human_command"
            "Human_command": "将容器B移动到容器C"},
        {"name":"takeout_items_by_human_command"
            "Human_command": "取出物品E"},
        {"name":"add_item_from_human_input"
            "Human_command": "放入物品F"},
        {"name":"query_item_from_human_query"   
            "Human_command": "询问物品G的位置"}
    ]
    用户的指令如下：###"""+human_command+"""
    ###
    请解读用户的指令，返回用户的操作列表和参数列表
    """
    functions=[
            {"name":"interpreter",
            "description": "interpret human command",
            "parameters":CommandList.schema()
            },
        ]
    function_call={"name":"interpreter"}


    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        functions=functions,
        function_call=function_call,
        temperature=0,
    )
    output = json.loads(response.choices[0]["message"]["function_call"]["arguments"])
    return output['commands']
