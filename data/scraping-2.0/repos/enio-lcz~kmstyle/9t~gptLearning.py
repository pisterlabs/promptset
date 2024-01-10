import numpy as np
import pandas as pd
import os
import io
import openai
import json
import inspect
import requests
import re


from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import base64
import email
from email import policy
from email.parser import BytesParser
from email.mime.text import MIMEText

from bs4 import BeautifulSoup
import dateutil.parser as parser

import json
import base64
from email.header import decode_header

import time

openai.api_key = os.getenv('OPENAI_API_KEY')
open_weather_key = os.getenv('OPENWEATHER_API_KEY')

def auto_functions(functions_list):
    """
    Chat模型的functions参数编写函数
    :param functions_list: 包含一个或者多个函数对象的列表；
    :return：满足Chat模型functions参数要求的functions对象
    """
    def functions_generate(functions_list):
        # 创建空列表，用于保存每个函数的描述字典
        functions = []

        def chen_ming_algorithm(data):
            """
            陈明算法函数，该函数定义了一种特殊的数据集计算过程
            :param data: 必要参数，表示带入计算的数据表，用字符串进行表示
            :return：陈明函数计算后的结果，返回结果为表示为JSON格式的Dataframe类型对象
            """
            df_new = pd.read_json(data)
            res = np.sum(df_new, axis=1) - 1
            return res.to_json(orient='records')
        
        chen_ming_function_description = inspect.getdoc(chen_ming_algorithm)
        
        chen_ming_function_name = chen_ming_algorithm.__name__
        
        chen_ming_function = {"name": "chen_ming_algorithm",
                              "description": "用于执行陈明算法的函数，定义了一种特殊的数据集计算过程",
                              "parameters": {"type": "object",
                                             "properties": {"data": {"type": "string",
                                                                     "description": "执行陈明算法的数据集"},
                                                           },
                                             "required": ["data"],
                                            },
                             }


        # 对每个外部函数进行循环
        for function in functions_list:
            # 读取函数对象的函数说明
            function_description = inspect.getdoc(function)
            # 读取函数的函数名字符串
            function_name = function.__name__


            user_message1 = '以下是某的函数说明：%s。' % chen_ming_function_description +\
                            '根据这个函数的函数说明，请帮我创建一个function对象，用于描述这个函数的基本情况。这个function对象是一个JSON格式的字典，\
                            这个字典有如下5点要求：\
                            1.字典总共有三个键值对；\
                            2.第一个键值对的Key是字符串name，value是该函数的名字：%s，也是字符串；\
                            3.第二个键值对的Key是字符串description，value是该函数的函数的功能说明，也是字符串；\
                            4.第三个键值对的Key是字符串parameters，value是一个JSON Schema对象，用于说明该函数的参数输入规范。\
                            5.输出结果必须是一个JSON格式的字典，只输出这个字典即可，前后不需要任何前后修饰或说明的语句' % chen_ming_function_name

            assistant_message1 = json.dumps(chen_ming_function)
            
            user_prompt = '现在有另一个函数，函数名为：%s；函数说明为：%s；\
                          请帮我仿造类似的格式为当前函数创建一个function对象。' % (function_name, function_description)

            response = openai.ChatCompletion.create(
                              model="gpt-3.5-turbo",
                              messages=[
                                {"role": "user", "name":"example_user", "content": user_message1},
                                {"role": "assistant", "name":"example_assistant", "content": assistant_message1},
                                {"role": "user", "name":"example_user", "content": user_prompt}]
                            )
            functions.append(json.loads(response.choices[0].message['content']))
        return functions
    
    max_attempts = 3
    attempts = 0

    while attempts < max_attempts:
        try:
            functions = functions_generate(functions_list)
            break  # 如果代码成功执行，跳出循环
        except Exception as e:
            attempts += 1  # 增加尝试次数
            print("发生错误：", e)
            if attempts == max_attempts:
                print("已达到最大尝试次数，程序终止。")
                raise  # 重新引发最后一个异常
            else:
                print("正在重新运行...")
    return functions


def get_weather(loc):
    """
    查询即时天气函数
    :param loc: 必要参数，字符串类型，用于表示查询天气的具体城市名称，\
    注意，中国的城市需要用对应城市的英文名称代替，例如如果需要查询北京市天气，则loc参数需要输入'Beijing'；
    :return：OpenWeather API查询即时天气的结果，具体URL请求地址为：https://api.openweathermap.org/data/2.5/weather\
    返回结果对象类型为解析之后的JSON格式对象，并用字符串形式进行表示，其中包含了全部重要的天气信息
    """
    # Step 1.构建请求
    url = "https://api.openweathermap.org/data/2.5/weather"

    # Step 2.设置查询参数
    params = {
        "q": loc,               
        "appid": open_weather_key,    # 输入API key
        "units": "metric",            # 使用摄氏度而不是华氏度
        "lang":"zh_cn"                # 输出语言为简体中文
    }

    # Step 3.发送GET请求
    response = requests.get(url, params=params)
    
    # Step 4.解析响应
    data = response.json()
    return json.dumps(data)

def run_conversation(messages, functions_list=None, model="gpt-3.5-turbo-16k"):
    """
    能够自动执行外部函数调用的Chat对话模型
    :param messages: 必要参数，字典类型，输入到Chat模型的messages参数对象
    :param functions_list: 可选参数，默认为None，可以设置为包含全部外部函数的列表对象
    :param model: Chat模型，可选参数，默认模型为gpt-4
    :return：Chat模型输出结果
    """
    # 如果没有外部函数库，则执行普通的对话任务
    if functions_list == None:
        response = openai.ChatCompletion.create(
                        model=model,
                        messages=messages,
                        )
        response_message = response["choices"][0]["message"]
        final_response = response_message["content"]
        
    # 若存在外部函数库，则需要灵活选取外部函数并进行回答
    else:
        # 创建functions对象
        functions = auto_functions(functions_list)
        # 创建外部函数库字典
        available_functions = {func.__name__: func for func in functions_list}

        # first response
        response = openai.ChatCompletion.create(
                        model=model,
                        messages=messages,
                        functions=functions,
                        function_call="auto")
        response_message = response["choices"][0]["message"]

        # 判断返回结果是否存在function_call，即判断是否需要调用外部函数来回答问题
        if response_message.get("function_call"):
            # 需要调用外部函数
            # 获取函数名
            function_name = response_message["function_call"]["name"]
            # 获取函数对象
            fuction_to_call = available_functions[function_name]
            # 获取函数参数
            function_args = json.loads(response_message["function_call"]["arguments"])
            # 将函数参数输入到函数中，获取函数计算结果
            function_response = fuction_to_call(**function_args)

            # messages中拼接first response消息
            messages.append(response_message)  
            # messages中拼接函数输出结果
            messages.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                }
            )  
            # 第二次调用模型
            second_response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
            )  
            # 获取最终结果
            final_response = second_response["choices"][0]["message"]["content"]
        else:
            final_response = response_message["content"]
    
    return final_response



def chat_with_model(prompt, functions_list=None,model="gpt-3.5-turbo-16k"):
    messages = [
        {"role": "system", "content": "你是一位机器学习工程师。"},
        {"role": "user", "content": prompt}
    ]
    
    while True:
        if functions_list is None:
            response = openai.ChatCompletion.create(
            model=model,
            messages=messages
            )
            # 获取模型回答
            answer = response.choices[0].message['content']
            print(f"模型回答: {answer}")
        else:
            res = run_conversation(messages=messages,functions_list=functions_list)
            print(f"模型回答: {res}") 
        

        # 询问用户是否还有其他问题
        user_input = input("您还有其他问题吗？(输入退出以结束对话): ")
        print(f"用户问:{user_input}")
        if user_input == "退出":
            break

        # 记录用户回答
        messages.append({"role": "user", "content": user_input})

def extract_code(s):
    """
    如果输入的字符串s是一个包含Python代码的Markdown格式字符串，提取出代码部分。
    否则，返回原字符串。

    参数:
    s: 输入的字符串。

    返回:
    提取出的代码部分，或原字符串。
    """
    # 判断字符串是否是Markdown格式
    if '```python' in s or 'Python' in s or'PYTHON' in s:
        # 找到代码块的开始和结束位置
        code_start = s.find('def')
        code_end = s.find('```\n', code_start)
        # 提取代码部分
        code = s[code_start:code_end]
    else:
        # 如果字符串不是Markdown格式，返回原字符串
        code = s

    return code



def get_recent_emails(n=5, userId='me'):
    """
    获取最近的n封邮件

    功能说明：
        此函数用于通过Gmail API获取指定用户的最近n封邮件信息，并将结果以包含多个字典的列表返回，以json格式进行表示。

    参数情况：
        n: 整数，代表需要查询的邮件个数
        userId: 字符串参数，默认情况下取值为'me'，表示查看我的邮件

    返回结果：
        返回一个包含多个字典的列表，并用json格式进行表示，其中一个字典代表一封邮件信息，每个字典中需要包含邮件的发件人、发件时间、邮件主题和邮件内容四个方面信息

    示例：
        get_recent_emails(5, 'me')
    """
    
    # 从token.json加载用户凭据
    creds = Credentials.from_authorized_user_file('token.json', ['https://www.googleapis.com/auth/gmail.readonly'])

    # 建立Gmail服务连接
    service = build('gmail', 'v1', credentials=creds)

    # 获取邮件列表
    results = service.users().messages().list(userId=userId, maxResults=n).execute()
    messages = results.get('messages', [])

    emails = []

    # 循环处理每封邮件
    for message in messages:
        msg = service.users().messages().get(userId=userId, id=message['id']).execute()

        # 获取邮件发件人、发件时间、邮件主题和邮件内容
        email_data = msg['payload']['headers']
        for values in email_data:
            name = values['name']
            if name == 'From':
                from_name = values['value']
            if name == 'Date':
                date = values['value']
            if name == 'Subject':
                subject, encoding = decode_header(values['value'])[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding if encoding else 'utf8')
        content = msg['snippet']

        # 将邮件信息添加到列表中
        emails.append({
            'from': from_name,
            'date': date,
            'subject': subject,
            'content': content
        })

    return json.dumps(emails, indent=4, ensure_ascii=False)


def extract_function_code(s, detail=0, tested=False, g=globals()):
    """
    函数提取函数，同时执行函数内容，可以选择打印函数信息，并选择代码保存的地址
    """
    def extract_code(s):
        """
        如果输入的字符串s是一个包含Python代码的Markdown格式字符串，提取出代码部分。
        否则，返回原字符串。

        参数:
        s: 输入的字符串。

        返回:
        提取出的代码部分，或原字符串。
        """
        # 判断字符串是否是Markdown格式
        if '```python' in s or 'Python' in s or'PYTHON' in s:
            # 找到代码块的开始和结束位置
            code_start = s.find('def')
            code_end = s.find('```\n', code_start)
            # 提取代码部分
            code = s[code_start:code_end]
        else:
            # 如果字符串不是Markdown格式，返回原字符串
            code = s

        return code
    
    # 提取代码字符串
    code = extract_code(s)
    
    # 提取函数名称
    match = re.search(r'def (\w+)', code)
    function_name = match.group(1)
    
    # 在untested文件夹内创建函数同名文件夹
    directory = './autoGmail_project/untested functions/%s' % function_name
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # 将函数写入本地
    if tested == False:
        with open('./autoGmail_project/untested functions/%s/%s_module.py' % (function_name, function_name), 'w', encoding='utf-8') as f:
            f.write(code)
    else:
        # 调用remove_to_test函数将函数文件夹转移至tested文件夹内
        remove_to_tested(function_name)
        with open('./autoGmail_project/tested functions/%s/%s_module.py' % (function_name, function_name), 'w', encoding='utf-8') as f:
            f.write(code)
    
    # 执行该函数
    try:
        exec(code, g)
    except Exception as e:
        print("An error occurred while executing the code:")
        print(e)
    
    # 打印函数名称
    if detail == 0:
        print("The function name is:%s" % function_name)
    
    if detail == 1:
        if tested == False:
            with open('./autoGmail_project/untested functions/%s/%s_module.py' % (function_name, function_name), 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            with open('./autoGmail_project/tested functions/%s/%s_module.py' % (function_name, function_name), 'r', encoding='utf-8') as f:   
                content = f.read()
                
        print(content)
        
    return function_name

def retrieve_emails(n, user_id='me'):
    """
    获取指定数量的最近邮件。

    参数:
    n: 要检索的邮件的数量。这应该是一个整数。
    user_id: 要检索邮件的用户的ID。默认值是'me'，表示当前授权的用户。

    返回:
    一个列表，其中每个元素都是一个字典，表示一封邮件。每个字典包含以下键：
    'From': 发件人的邮箱地址。
    'Date': 邮件的发送日期。
    'Subject': 邮件的主题。
    'Snippet': 邮件的摘要（前100个字符）。
    """
    # 从本地文件中加载凭据
    creds = Credentials.from_authorized_user_file('token.json')

    # 创建 Gmail API 客户端
    service = build('gmail', 'v1', credentials=creds)

    # 获取邮件列表
    results = service.users().messages().list(userId=user_id).execute()
    messages = results.get('messages', [])[:n]

    emails = []
    for message in messages:
        msg = service.users().messages().get(userId=user_id, id=message['id']).execute()

        # 解码邮件内容
        payload = msg['payload']
        headers = payload.get("headers")
        parts = payload.get("parts")

        data = {}

        if headers:
            for d in headers:
                name = d.get("name")
                if name.lower() == "from":
                    data['From'] = d.get("value")
                if name.lower() == "date":
                    data['Date'] = parser.parse(d.get("value")).strftime('%Y-%m-%d %H:%M:%S')
                if name.lower() == "subject":
                    data['Subject'] = d.get("value")

        if parts:
            for part in parts:
                mimeType = part.get("mimeType")
                body = part.get("body")
                data_decoded = base64.urlsafe_b64decode(body.get("data")).decode()
                if mimeType == "text/plain":
                    data['Snippet'] = data_decoded
                elif mimeType == "text/html":
                    soup = BeautifulSoup(data_decoded, "html.parser")
                    data['Snippet'] = soup.get_text()

        emails.append(data)

    # 返回邮件列表
    return json.dumps(emails, indent=2)

def get_email_count(userId='me'):
    """
    查询Gmail邮箱中的邮件数量
    :param userId: 必要参数，字符串类型，用于表示需要查询的邮箱ID，\
    注意，当查询我的邮箱时，userId需要输入'me'；
    :return：包含邮件数量的对象，该对象由Gmail API返回，并以JSON格式保存。
    """
    # 从本地文件中加载凭据
    creds = Credentials.from_authorized_user_file('token.json')
    
    # 创建 Gmail API 客户端
    service = build('gmail', 'v1', credentials=creds)
    
    # 获取邮箱中的邮件数量
    results = service.users().messages().list(userId=userId, maxResults=1).execute()
    total_count = results['resultSizeEstimate']
    
    # 将结果保存为JSON格式对象
    result_object = {'email_count': total_count}
    
    return json.dumps(result_object)

def send_email(to, subject, message_text):
    """
    借助Gmail API创建并发送邮件函数
    :param to: 必要参数，字符串类型，用于表示邮件发送的目标邮箱地址；
    :param subject: 必要参数，字符串类型，表示邮件主题；
    :param message_text: 必要参数，字符串类型，表示邮件全部正文；
    :return：返回发送结果字典，若成功发送，则返回包含邮件ID和发送状态的字典。
    """
    
    creds_file='token_send.json'
    
    def create_message(to, subject, message_text):
        """创建一个MIME邮件"""
        message = MIMEText(message_text)
        message['to'] = to
        message['from'] = 'me'
        message['subject'] = subject
        raw_message = base64.urlsafe_b64encode(message.as_string().encode('utf-8')).decode('utf-8')
        return {
            'raw': raw_message
        }

    def send_message(service, user_id, message):
        """发送邮件"""
        try:
            sent_message = service.users().messages().send(userId=user_id, body=message).execute()
            print(f'Message Id: {sent_message["id"]}')
            return sent_message
        except Exception as e:
            print(f'An error occurred: {e}')
            return None

    # 从本地文件中加载凭据
    creds = Credentials.from_authorized_user_file(creds_file)

    # 创建 Gmail API 客户端
    service = build('gmail', 'v1', credentials=creds)

    message = create_message(to, subject, message_text)
    res = send_message(service, 'me', message)

    return json.dumps(res)

def get_latest_email(userId):
    """
    查询Gmail邮箱中最后一封邮件信息
    :param userId: 必要参数，字符串类型，用于表示需要查询的邮箱ID，\
    注意，当查询我的邮箱时，userId需要输入'me'；
    :return：包含最后一封邮件全部信息的对象，该对象由Gmail API创建得到
    """
    # 从本地文件中加载凭据
    creds = Credentials.from_authorized_user_file('token.json')
    
    # 创建 Gmail API 客户端
    service = build('gmail', 'v1', credentials=creds)
    
    # 列出用户的一封最新邮件
    results = service.users().messages().list(userId=userId, maxResults=1).execute()
    messages = results.get('messages', [])

    # 遍历邮件
    for message in messages:
        # 获取邮件的详细信息
        msg = service.users().messages().get(userId='me', id=message['id']).execute()
        
    return json.dumps(msg)

def remove_to_tested(function_name):
    """
    将函数同名文件夹由untested文件夹转移至tested文件夹内。\
    完成转移则说明函数通过测试，可以使用。此时需要将该函数的源码写入gptLearning.py中方便下次调用。
    """
    
    # 将函数代码写入gptLearning.py文件中
    with open('./autoGmail_project/untested functions/%s/%s_module.py' % (function_name, function_name), encoding='utf-8') as f:
        function_code = f.read()
    
    with open('gptLearning.py', 'a', encoding='utf-8') as f:
        f.write(function_code)
    
    # 源文件夹路径
    src_dir = './autoGmail_project/untested functions/%s' % function_name

    # 目标文件夹路径
    dst_dir = './autoGmail_project/tested functions/%s' % function_name
    
    # 移动文件夹
    shutil.move(src_dir, dst_dir)



def show_functions(tested=False, if_print=False):
    """
    打印tested或untested文件夹内全部函数
    """
    current_directory = os.getcwd()
    if tested == False:
        directory = current_directory + '/autoGmail_project/untested functions/'
    else:
        directory = current_directory + '/autoGmail_project/tested functions/'
    files_and_directories = os.listdir(directory)
    # 过滤结果，只保留.py文件和非__pycache__文件夹
    files_and_directories = files_and_directories = [name for name in files_and_directories if (os.path.splitext(name)[1] == '.py' or os.path.isdir(os.path.join(directory, name))) and name != "__pycache__"]
    
    if if_print != False:
        for name in files_and_directories:
            print(name)
    
    return files_and_directories

def get_decompose_results(req, model='gpt-3.5-turbo-16k'):
    """
    复杂需求拆解函数，能够将用户输入的复杂需求拆解为一系列更容易完成的子任务
    :param req: 必选参数，以字符串形式表示，用于表示用户输入的原始需求；
    :param model: 拆解需求所使用的大模型；
    :return：由子任务所组成的列表；
    """
    
    decompose_results = []
    
    with open('./autoGmail_project/tested functions/decompose_messages.json', 'r') as f:
        decompose_messages = json.load(f)
        
    decompose_messages.append({"role": "user", "content": req})
    
    response = openai.ChatCompletion.create(
                model=model,
                messages=decompose_messages
            )
    
    res = response.choices[0].message['content']
    
    # 使用正则表达式查找以1、2、3开始的句子
    matches = re.findall(r'\d\.(.*?。)', res)

    for match in matches:
        decompose_results.append(match)
        
    return decompose_results

def code_generate(req, few_shot='all', model='gpt-3.5-turbo-16k', g=globals(), detail=0, system_messages=None):
    """
    Function calling外部函数自动创建函数，可以根据用户的需求，直接将其翻译为Chat模型可以直接调用的外部函数代码。
    :param req: 必要参数，字符串类型，表示输入的用户需求；
    :param few_shot: 可选参数，默认取值为字符串all，用于描述Few-shot提示示例的选取方案，当输入字符串all时，则代表提取当前外部函数库中全部测试过的函数作为Few-shot；\
    而如果输入的是一个包含了多个函数名称的list，则表示使用这些函数作为Few-shot。
    :param model: 可选参数，表示调用的Chat模型，默认选取gpt-4-0613；
    :param g: 可选参数，表示extract_function_code函数作用域，默认为globals()，即在当前操作空间全域内生效；
    :param detail: 可选参数，默认取值为0，还可以取值为1，表示extract_function_code函数打印新创建的外部函数细节；
    :param system_messages: 可选参数，默认取值为None，此时读取本地system_messages.json作为系统信息，也可以自定义；
    :return：新创建的函数名称。需要注意的是，在函数创建时，该函数也会在当前操作空间被定义，后续可以直接调用；
    """
    
    # 提取提示示例的函数名称
    if few_shot == 'all':
        few_shot_functions_name = show_functions(tested=True)
    elif type(few_shot) == list:
        few_shot_functions_name = few_shot
    # few_shot_functions = [globals()[name] for name in few_shot_functions_name]
    
    # 读取各阶段系统提示
    if system_messages == None:
        with open('./autoGmail_project/tested functions/system_messages.json', 'r') as f:
            system_messages = json.load(f)
        
    # 各阶段提示message对象
    few_shot_messages_CM = []
    few_shot_messages_CD = []
    few_shot_messages = []
    
    # 先保存第一条消息，也就是system message
    few_shot_messages_CD += system_messages["system_message_CD"]
    few_shot_messages_CM += system_messages["system_message_CM"]
    few_shot_messages += system_messages["system_message"]

    # 创建不同阶段提示message
    for function_name in few_shot_functions_name:
        with open('./autoGmail_project/tested functions/%s/%s_prompt.json' % (function_name, function_name), 'r') as f:
            msg = json.load(f)
        few_shot_messages_CD += msg["stage1_CD"]
        few_shot_messages_CM += msg["stage1_CM"]
        few_shot_messages += msg['stage2']
        
    # 读取用户需求，作为第一阶段CD环节User content
    new_req_CD_input = req
    few_shot_messages_CD.append({"role": "user", "content": new_req_CD_input})
    
    print('第一阶段CD环节提示创建完毕，正在进行CD提示...')
    
    # 第一阶段CD环节Chat模型调用过程
    response = openai.ChatCompletion.create(
                  model=model,
                  messages=few_shot_messages_CD
                )
    new_req_pi = response.choices[0].message['content']
    
    print('第一阶段CD环节提示完毕')
    
    # 第一阶段CM环节Messages创建
    new_req_CM_input = new_req_CD_input + new_req_pi
    few_shot_messages_CM.append({"role": "user", "content": new_req_CM_input})
    
    print('第一阶段CM环节提示创建完毕，正在进行第一阶段CM提示...')
    # 第一阶段CM环节Chat模型调用过程
    response = openai.ChatCompletion.create(
                      model=model,
                      messages=few_shot_messages_CM
                    )
    new_req_description = response.choices[0].message['content']
    
    print('第一阶段CM环节提示完毕')
    
    # 第二阶段Messages创建过程
    few_shot_messages.append({"role": "user", "content": new_req_description})
    
    print('第二阶段提示创建完毕，正在进行第二阶段提示...')
    
    # 第二阶段Chat模型调用过程
    response = openai.ChatCompletion.create(
                  model=model,
                  messages=few_shot_messages
                )
    new_req_function = response.choices[0].message['content']
    
    print('第二阶段提示完毕，准备运行函数并编写提示示例')
    
    # 提取函数并运行，创建函数名称对象，统一都写入untested文件夹内
    function_name = extract_function_code(s=new_req_function, detail=detail, g=g)
    
    print('新函数保存在./autoGmail_project/untested functions/%s/%s_module.py文件中' % (function_name, function_name))
    
    # 创建该函数提示示例
    new_req_messages_CD = [
                          {"role": "user", "content": new_req_CD_input},
                          {"role": "assistant", "content": new_req_pi}
                         ]
    new_req_messages_CM = [
                          {"role": "user", "content": new_req_CM_input},
                          {"role": "assistant", "content":new_req_description}
                         ]
    
    with open('./autoGmail_project/untested functions/%s/%s_module.py' % (function_name, function_name), encoding='utf-8') as f:
        new_req_function = f.read()
    
    new_req_messages = [
                       {"role": "user", "content": new_req_description},
                       {"role": "assistant", "content":new_req_function}
                      ] 
    
    new_req_prompt = {
                     "stage1_CD": new_req_messages_CD,
                     "stage1_CM": new_req_messages_CM,
                     "stage2": new_req_messages
                    }   
    
    with open('./autoGmail_project/untested functions/%s/%s_prompt.json' % (function_name, function_name), 'w') as f:
        json.dump(new_req_prompt, f)
        
    print('新函数提示示例保存在./autoGmail_project/untested functions/%s/%s_prompt.json文件中' % (function_name, function_name))
    print('done')
    return function_name


def prompt_modified(function_name, system_content='推理链修改.md', model="gpt-4-0613", g=globals(), system_messages=None):
    """
    智能邮件项目的外部函数审查函数，用于审查外部函数创建流程提示是否正确以及最终创建的代码是否正确
    :param function_name: 必要参数，字符串类型，表示审查对象名称；
    :param system_content: 可选参数，默认取值为字符串推理链修改.md，表示此时审查函数外部挂载文档名称，需要是markdwon格式文档；
    :param model: 可选参数，表示调用的Chat模型，默认选取gpt-4-0613；
    :param g: 可选参数，表示extract_function_code函数作用域，默认为globals()，即在当前操作空间全域内生效；
    :param system_messages: 可选参数，默认取值为None，此时读取本地system_messages.json作为系统信息，也可以自定义；
    :return：审查结束后新创建的函数名称
    """
    print("正在执行审查函数，审查对象：%s" % function_name)
    
    if system_messages != None:
        system_content='复杂需求推理链修改.md'
        
    with open(system_content, 'r', encoding='utf-8') as f:
        md_content = f.read()  
        
    # 读取原函数全部提示内容
    with open('./autoGmail_project/untested functions/%s/%s_prompt.json' % (function_name, function_name), 'r') as f:
        msg = json.load(f)
    
    # 将其保存为字符串
    msg_str = json.dumps(msg)
    
    # 进行审查
    response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                    {"role": "system", "content": md_content},
                    {"role": "user", "content": '以下是一个错误的智能邮件项目的推理链，请你按照要求对其进行修改：%s' % msg_str}
                    ]
                )
    
    modified_result = response.choices[0].message['content']
    
    def extract_json(s):
        pattern = r'```[jJ][sS][oO][nN]\s*({.*?})\s*```'
        match = re.search(pattern, s, re.DOTALL)
        if match:
            return match.group(1)
        else:
            return s
    
    modified_json = extract_json(modified_result)
    
    # 提取函数源码
    code = json.loads(modified_json)['stage2'][1]['content']
    
    # 提取函数名
    match = re.search(r'def (\w+)', code)
    function_name = match.group(1)
    
    print("审查结束，新的函数名称为：%s。\n正在运行该函数定义过程，并保存函数源码与prompt" % function_name)
    
    exec(code, g)
    
    # 在untested文件夹内创建函数同名文件夹
    directory = './autoGmail_project/untested functions/%s' % function_name
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    # 写入函数
    with open('./autoGmail_project/untested functions/%s/%s_module.py' % (function_name, function_name), 'w', encoding='utf-8') as f:
        f.write(code)
        
    # 写入提示
    with open('./autoGmail_project/untested functions/%s/%s_prompt.json' % (function_name, function_name), 'w') as f:
        json.dump(json.loads(modified_json), f)
    
    print('新函数提示示例保存在./autoGmail_project/untested functions/%s/%s_prompt.json文件中' % (function_name, function_name))
    print("%s函数已在当前操作空间定义，可以进行效果测试" % function_name)
    
    return function_name

def get_email_info(userId='me', n=1):
    """
    获取指定邮箱的第一封邮件的信息。

    参数：
    userId: 要查询的邮箱的用户ID，默认为'me'，表示当前授权的用户。
    n: 要查询的邮件的索引，默认为1，表示查询第一封邮件。

    返回：
    一个字典，包含邮件的收件时间、发送人和邮件内容的键值对，键包括 '收件时间'、'发送人' 和 '邮件内容'。
    """
    # 从本地文件中加载凭据
    creds = Credentials.from_authorized_user_file('token.json')

    # 创建 Gmail API 客户端
    service = build('gmail', 'v1', credentials=creds)

    # 查询邮件列表
    results = service.users().messages().list(userId=userId).execute()
    messages = results.get('messages', [])[:n]

    email_info = {}

    if messages:
        message = messages[0]
        msg = service.users().messages().get(userId=userId, id=message['id']).execute()

        # 解码邮件内容
        payload = msg['payload']
        headers = payload.get("headers")
        parts = payload.get("parts")

        if headers:
            for header in headers:
                name = header.get("name")
                if name.lower() == "from":
                    email_info['发送人'] = header.get("value")
                if name.lower() == "date":
                    email_info['收件时间'] = parser.parse(header.get("value")).strftime('%Y-%m-%d %H:%M:%S')

        if parts:
            for part in parts:
                mimeType = part.get("mimeType")
                body = part.get("body")
                if mimeType == "text/plain":
                    data_decoded = base64.urlsafe_b64decode(body.get("data")).decode()
                    email_info['邮件内容'] = data_decoded
                elif mimeType == "text/html":
                    data_decoded = base64.urlsafe_b64decode(body.get("data")).decode()
                    soup = BeautifulSoup(data_decoded, "html.parser")
                    email_info['邮件内容'] = soup.get_text()

    return json.dumps(email_info)


def function_test(function_name, req, few_shot, model="gpt-4-0613", g=globals(), system_messages=None):

    def test_messages(ueser_content):
        messages = [{"role": "system", "content": "端木天的邮箱地址是:2323365771@qq.com"},
                    {"role": "system", "content": "我的邮箱地址是:ksken166@gmail.com"},
                    {"role": "user", "content": ueser_content}]
        return messages
            
    messages = test_messages(req)
    
    new_function = globals()[function_name]
    functions_list = [new_function]
    
    print("根据既定用户需求req进行%s函数功能测试，请确保当该函数已经在当前操作空间定义..." % function_name)
    
    # 有可能在run_conversation环节报错
    # 若没报错，则运行：
    try:
        final_response = run_conversation(messages=messages, functions_list=functions_list, model=model)
        print("当前函数运行结果：'%s'" % final_response)
        
        feedback = input("函数功能是否满足要求 (yes/no)? ")
        if feedback.lower() == 'yes':
            print("函数功能通过测试，正在将函数写入tested文件夹")
            remove_to_tested(function_name)
            print('done')
        else:
            next_step = input("函数功能未通过测试，是1.需要再次进行测试，还是2.进入debug流程？")
            if next_step == '1':
                print("准备再次测试...")
                function_test(function_name=function_name, req=req, few_shot=few_shot, g=g, system_messages=system_messages)
            else:
                solution = input("请选择debug方案：\n1.再次执行函数创建流程，并测试结果；\n2.执行审查函数\
                \n3.重新输入用户需求；\n4.退出程序，进行手动尝试")
                if solution == '1':
                    # 再次运行函数创建过程
                    print("好的，正在尝试再次创建函数，请稍等...")
                    few_shot_str = input("准备再次测试，请问是1.采用此前Few-shot方案，还是2.带入全部函数示例进行Few-shot？")
                    if few_shot_str == '1':
                        Gmail_auto_func(req=req, few_shot=few_shot, model=model, g=g)
                    else:
                        Gmail_auto_func(req=req, few_shot='all', model=model, g=g)
                elif solution == '2':
                    # 执行审查函数
                    print("好的，执行审查函数，请稍等...")
                    function_name = prompt_modified(function_name=function_name, model="gpt-3.5-turbo-16k-0613", g=g, system_messages=system_messages)
                    # 接下来带入进行测试
                    print("新函数已创建，接下来带入进行测试...")
                    function_test(function_name=function_name, req=req, few_shot=few_shot, g=g, system_messages=system_messages)
                elif solution == '3':
                    new_req = input("好的，请再次输入用户需求，请注意，用户需求描述方法将极大程度影响最终函数创建结果。")
                    few_shot_str = input("接下来如何运行代码创建函数？1.采用此前Few-shot方案；\n2.使用全部外部函数作为Few-shot")
                    if few_shot_str == '1':
                        Gmail_auto_func(req=req, few_shot=few_shot, model=model, g=g)
                    else:
                        Gmail_auto_func(req=req, few_shot='all', model=model, g=g)
                elif solution == '4':
                    print("好的，预祝debug顺利~")
        
    # run_conversation报错时则运行：
    except Exception as e:
        next_step = input("run_conversation无法正常运行，接下来是1.再次运行运行run_conversation，还是2.进入debug流程？")
        if next_step == '1':
            function_test(function_name=function_name, req=new_req, few_shot=few_shot, g=g, system_messages=system_messages)
        else:
            solution = input("请选择debug方案：\n1.再次执行函数创建流程，并测试结果；\n2.执行审查函数\
            \n3.重新输入用户需求；\n4.退出程序，进行手动尝试")
            if solution == '1':
                # 再次运行函数创建过程
                print("好的，正在尝试再次创建函数，请稍等...")
                few_shot_str = input("准备再次测试，请问是1.采用此前Few-shot方案，还是2.带入全部函数示例进行Few-shot？")
                if few_shot_str == '1':
                    Gmail_auto_func(req=req, few_shot=few_shot, model=model, g=g)
                else:
                    Gmail_auto_func(req=req, few_shot='all', model=model, g=g)
            elif solution == '2':
                # 执行审查函数
                print("好的，执行审查函数，请稍等...")
                max_attempts = 3
                attempts = 0

                while attempts < max_attempts:
                    try:
                        function_name = prompt_modified(function_name=function_name, model="gpt-3.5-turbo-16k-0613", g=g, system_messages=system_messages)
                        break  # 如果代码成功执行，跳出循环
                    except Exception as e:
                        attempts += 1  # 增加尝试次数
                        print("发生错误：", e)
                        if attempts == max_attempts:
                            print("已达到最大尝试次数，程序终止。")
                            raise  # 重新引发最后一个异常
                        else:
                            print("正在重新运行审查程序...")
                # 接下来带入进行测试
                print("新函数已创建，接下来带入进行测试...")
                function_test(function_name=function_name, req=req, few_shot=few_shot, g=g, system_messages=system_messages)
            elif solution == '3':
                new_req = input("好的，请再次输入用户需求，请注意，用户需求描述方法将极大程度影响最终函数创建结果。")
                few_shot_str = input("接下来如何运行代码创建函数？1.采用此前Few-shot方案；\n2.使用全部外部函数作为Few-shot")
                if few_shot_str == '1':
                    Gmail_auto_func(req=req, few_shot=few_shot, model=model, g=g)
                else:
                    Gmail_auto_func(req=req, few_shot='all', model=model, g=g)
            elif solution == '4':
                print("好的，预祝debug顺利~")


def Gmail_auto_func(req, few_shot='all', model='gpt-3.5-turbo-16k', g=globals(), detail=0):
    # 默认情况下system_messages = None
    system_messages = None
    
    # 尝试进行任务拆解
    try:
        decompose_results = get_decompose_results(req=req, model=model)
    except Exception as e:
        print(e)
        print('暂停1分钟后继续调用模型')
        time.sleep(60)
        decompose_results = get_decompose_results(req=req, model=model)
        
    # 如果只拆解得到多个任务，则创建新的基于任务拆解得到的system_message
    if len(decompose_results) != 1:
        print('原始需求将拆分为多个子需求并进行分段代码创建与合并')
        # 读取原始system_message
        with open('./autoGmail_project/tested functions/system_messages.json', 'r') as f:
            system_messages = json.load(f)
            
        # 用于存储全部需求的函数代码
        sub_func_all = ''
        # 用于存储全部需求
        sub_req_all = ''
        # 计数器
        i = 0
        
        for sub_req in decompose_results:
            i += 1
            # 每个需求依次创建子函数
            print('第%s个子需求为：%s' % (i, sub_req))
            print('正在创建对应子函数')
            try:
                sub_func_name = code_generate(sub_req, few_shot=few_shot, g=g, detail=detail, model=model)
            except Exception as e:
                print(e)
                print('暂停1分钟后继续调用模型')
                time.sleep(60)
                sub_func_name = code_generate(sub_req, few_shot=few_shot, g=g, detail=detail, model=model)
                
            # 读取子函数源码
            with open('./autoGmail_project/untested functions/%s/%s_module.py' % (sub_func_name, sub_func_name), encoding='utf-8') as f:
                sub_func_str = f.read()
            # 对子函数源码进行拼接
            sub_func_all += sub_func_str
            
            # 按顺序拼接子需求
            sub_req_all += str(i)+('.')
            sub_req_all += sub_req            
        
        print('子需求对应的子函数全部创建完毕，接下来进入到原始需求函数创建过程...')
        # 添加一个system_message
        decompose_description = '对于当前编程需求，可以拆解为若干个子需求，也就是：%s。这些子需求的实现方式可以参考如下代码：%s' % (sub_req_all, sub_func_all)
        system_messages['system_message'].append({'role': 'system', 'content': decompose_description})
    
    # 进行代码创建和代码审查
    try:
        function_name = code_generate(req=req, few_shot=few_shot, model=model, g=g, detail=detail, system_messages=system_messages)
    except Exception as e:
        print(e)
        print('暂停1分钟后继续调用模型')
        time.sleep(60)
        function_name = code_generate(req=req, few_shot=few_shot, model=model, g=g, detail=detail, system_messages=system_messages)
    function_test(function_name=function_name, req=req, few_shot=few_shot, model=model, g=g, system_messages=system_messages)