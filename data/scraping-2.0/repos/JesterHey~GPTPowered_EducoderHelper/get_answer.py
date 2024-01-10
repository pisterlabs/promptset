'''
本模块用于答案获得
1、先获得本地的json文件
2、云端检查json是否存在，存在则直接调用，不存在则步骤3
3、调用openai的api获得答案并处理格式，生成新的json文件并存储在云端
'''
from openai import AsyncOpenAI
import os
import json
import asyncio
from cloud import download,delete
import base64
#获得指定目录下的所有数字开头的json文件
def get_shixunjson(file:str) -> list:
    '''
    file:文件夹路径
    '''
    files = os.listdir(file)
    jsonfiles = []
    for i in files:
        if i.endswith('.json') and i[0].isdigit():
            jsonfiles.append(i)
    return jsonfiles
#获得指定目录下的所有pro开头的json文件
def get_programmingjson(file:str) -> list:
    '''
    file:文件夹路径
    '''
    files = os.listdir(file)
    jsonfiles = []
    for i in files:
        if i.endswith('.json') and i.startswith('pro'):
            jsonfiles.append(i)
    return jsonfiles

'''
与云服务器连接，先判断当前json是否已在云服务器上，如果在，则直接调用，
节省调用API的时间和资费，否则，调用API，获得答案，并将答案存入云服务器
'''

# 以下封装成函数
# 重写本地json文件
def rewrite_shixun_json(json_name:str,new_data:dict):
    with open(json_name,'w',encoding='utf-8') as f:
        json.dump(new_data,f,ensure_ascii=False)

def rewrite_programming_json(json_names:list,new_data:list):
    '''
    函数用于把new_data中的答案写入对应的json中
    json_names:本地所有pro开头的json文件名
    new_data:与json_names中每个文件对应的答案所构成的列表
    '''
    # 检查json_names和new_data元素数量是否一致
    if len(json_names) == len(new_data):
        i=0
        while i< len(json_names):
            with open(json_names[i],'w',encoding='utf-8') as f:
                json.dump(new_data[i],f,ensure_ascii=False)
            i+=1


#读取json文件并转换为字典
def load_json_data(json_name:str) -> dict:
    with open(json_name,'r',encoding='utf-8') as f: # json_name为无答案的json文件名
        data = json.load(f)
    return data
def load_api_key() -> str:
    with open('apis.json','r',encoding='utf-8') as f: # apis.json为存储api_key的json文件名
        return json.load(f)['my_api']

#遍历字典，获得每一关的参数，构造请求，获得答案
'''
用于构造请求的参数：describe,require,code
向GPT提问的格式：promot + 参数模板化的问题
'''

#promot1 = f'现在，我想让你扮演一个资深而经验丰富的程序员来解一个问题，我的问题将由三个部分组成，第一部分是问题的描述，第二部分是问题的需求，第三部分是问题的代码，我需要你按照我的模板编写代码，使用的代码语言是{language}。并且你返回的代码应当是带有注释的。再次注意，请返回完整的，格式正确的，可读的代码！'
#promot2 = f'现在，我想让你扮演一个资深而经验丰富的程序员来解一个问题，我的问题会有两个部分组成，第一部分是问题的描述，第二部分是你需要补全或者完善的代码。你需要阅读，理解我的问题描述，然后补全或者完善代码,使用的代码语言是{language}。再次注意，请返回完整的，格式正确的，输入由用户给出的，可读的代码！'
#构造问题模板
#遍历字典，获得每一关的参数，构造请求，获得答案
#使用异步函数提升效率
'''
异步思路：由于每一关都会的答案查询都是独立的，可以把不同的查询请求构建成异步任务，谁先完成就先返回谁的答案，
最后把所有的答案整合到一个字典中，再写入json文件中
'''

# 初始化异步客户端
client = AsyncOpenAI(
    api_key=load_api_key(),
    base_url='https://api.op-enai.com/v1'
)
def get_shixunanswer_from_api(jsonfile:dict,client:AsyncOpenAI,promot:str) -> dict:
    '''
    jsonfile:本地json文件
    client:异步客户端
    promot:问题模板
    '''
    j_shixun = get_shixunjson(os.getcwd())
    j_programming = get_programmingjson(os.getcwd())
    if j_shixun == []:
        language = get_programmingjson(os.getcwd())[0].split('.')[0].split('_')[-1]
    else:
        language = get_shixunjson(os.getcwd())[0].split('.')[0].split('_')[-1]
    data = jsonfile
    promot1 = f'现在，我想让你扮演一个资深而经验丰富的程序员来解一个问题，我的问题将由三个部分组成，第一部分是问题的描述，第二部分是问题的需求，第三部分是问题的代码，我需要你按照我的模板编写代码，使用的代码语言是{language}。并且你返回的代码应当是带有注释的。再次注意，请返回完整的，格式正确的，可读的代码！并且，在没有特殊要求的情况下，不用给出用于运行补全后代码的示例代码'
    # 异步函数来获取答案
    async def get_answer(key,value) -> str:
        '''
        key:关卡id
        value:关卡参数
        '''
        cid = key
        # code 是base64编码的字符串，需要解码
        des, req, code = value['describe'], value['require'], base64.b64decode(value['code']).decode('utf-8')
        question = f'问题描述：{des}\n任务需求：{req}\n根据上面的需求，以下是你需要补充并完善代码：\n{code}'
        try:
            response = await client.chat.completions.create(
                model='gpt-4-1106-preview',
                messages=[
                    {'role': 'system', 'content': promot1},
                    {'role': 'user', 'content': question}
                ]
            )
            return f'{cid}/{response.choices[0].message.content}'
        except Exception as e:
            print(f'错误信息：{e}')

    # 主函数
    async def main(data) -> dict:
        ansewer_data = data
        tasks = [get_answer(cid,value) for cid,value in data.items()]
        answers = await asyncio.gather(*tasks) # 返回一个列表，列表中的每个元素为每个异步任务的返回值
        #由于异步获得的答案顺序不确定，需要处理,先把答案按照关卡id排序
        answers = sorted(answers,key=lambda x:int(x.split('/')[0]))
        # 在data的每个value中新增一个键值对，键为answer，值为答案，并作为返回值返回
        for i in range(len(answers)):
            ansewer_data[list(ansewer_data.keys())[i]]['answer'] = answers[i].split('/')[-1]

        return ansewer_data


    # 运行主函数
    return asyncio.run(main(data=data))


# 由于编程作业会涉及到多个文件,整合为一个文件池，文件池中的每个json就是异步的最小任务单元
# 这样可以多个文件请求并发，异步协程，提升效率
def get_programming_answer_from_api(jsonfile:list,client:AsyncOpenAI,promot:str) -> list:
    '''
    jsonfile:本地json文件
    client:异步客户端
    promot:问题模板
    '''
    j_shixun = get_shixunjson(os.getcwd())
    j_programming = get_programmingjson(os.getcwd())
    if j_shixun == []:
        language = get_programmingjson(os.getcwd())[0].split('.')[0].split('_')[-1]
    else:
        language = get_shixunjson(os.getcwd())[0].split('.')[0].split('_')[-1]
    data = jsonfile
    promot2 = f'现在，我想让你扮演一个资深而经验丰富的程序员来解一个问题，我的问题会有两个部分组成，第一部分是问题的描述，第二部分是你需要补全或者完善的代码。你需要阅读，理解我的问题描述，然后补全或者完善代码,使用的代码语言是{language}。再次注意，请返回完整的，格式正确的，输入由用户给出的，可读的代码！并且，在没有特殊要求的情况下，不用给出用于运行补全后代码的示例代码'
    # 异步函数来获取答案
    async def get_answer(value:dict) -> str:
        pro_id = value['id']
        # code 是base64编码的字符串，需要解码
        des,code = value['describe'],base64.b64decode(value['code']).decode('utf-8')
        question = f'问题描述：{des}\n根据上面的需求，以下是你需要补充并完善代码：\n{code}'
        try:
            response = await client.chat.completions.create(
                model='gpt-4-1106-preview',
                #model = 'gpt-3.5-turbo',
                messages=[
                    {'role': 'system', 'content': promot2},
                    {'role': 'user', 'content': question}
                ]
            )
            return f'{pro_id}/{response.choices[0].message.content}'
        except Exception as e:
            print(f'错误信息：{e}')


    # 主函数
    async def main(datalist:list) -> list:
        # 由于编程作业会涉及到多个文件,整合为一个文件池
        '''
        data:本地json文件池
        '''
        # 把data中的每个json文件读取为字典
        datalist =  [load_json_data(i) for i in datalist]
        # 把每个字典扔给get_answer函数，获得答案,异步获取信息
        tasks = [get_answer(value) for value in datalist]
        answers = await asyncio.gather(*tasks) # answers是一个列表，列表中的每个元素为每个异步任务的返回值
        # 由于异步获得的答案顺序不确定，需要处理,先把答案按照pro_id排序
        answers = sorted(answers,key=lambda x:x.split('/')[0])
        # 在datalist中的每个字典中新增一个键值对，键为answer，值为答案，并作为返回值返回
        for i in range(len(answers)):
            datalist[i]['answer'] = answers[i].split('/')[-1]
        # 返回datalist
        return datalist

    # 运行主函数,返回一个列表，列表中的每个元素为每个异步任务的返回值，即重写的字典
    return asyncio.run(main(datalist=data))




if __name__ == '__main__':
    promot=''
    ans = get_programming_answer_from_api(jsonfile=get_programmingjson(os.getcwd()),client=client,promot=promot)
    print(ans)
    rewrite_programming_json(json_names=get_programmingjson(os.getcwd()),new_data=ans)
