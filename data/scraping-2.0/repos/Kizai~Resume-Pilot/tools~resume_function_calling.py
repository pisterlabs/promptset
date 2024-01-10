import os
import json
import langchain
import openai
from dotenv import load_dotenv
from tools.pdfreader import PdfContentReader
from tools.logginger import get_logger
from config import API_KEY

# 设置logger
logger = get_logger()

# 从环境变量中加载OpenAI的API秘钥
load_dotenv()
openai.api_key = API_KEY

# 实例化PDF阅读器
pdf_reader = PdfContentReader()

# 用于描述函数的对象
function_descriptions = [
    {
        "name": "does_resume_meet_requirements",
        "description": "判断简历是否符合招聘需求",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "如张三",
                },
                "email": {
                    "type": "string",
                    "description": "如xiaoming@gmail.com",
                },
                "is_match": {
                    "type": "boolean",
                    "description": "True or False",
                },
                "reason": {
                    "type": "string",
                    "description": "该简历符合岗位的全部需求或该求职者的期望城市不是深圳所以不符合岗位需求。"
                }
            },
            "required": ["name", "email", "is_match", "reason"],
        },
    }
]


def extract_information_from_response(response):
    """
    从OpenAI的响应中提取信息。

    Args:
        response: OpenAI的响应。

    Returns:
        name: 姓名。
        email: 邮箱。
        is_match: 是否匹配。
        reason: 原因。
    """
    function_call_arguments = json.loads(response['function_call']['arguments'])
    name = function_call_arguments.get("name")
    email = function_call_arguments.get("email")
    is_match = function_call_arguments.get("is_match")
    reason = function_call_arguments.get("reason")

    return name, email, is_match, reason


def get_resume_meet_requirements(name, email, is_match, reason):
    """
    获取简历是否满足要求的信息。

    Args:
        name: 姓名。
        email: 邮箱。
        is_match: 是否匹配。
        reason: 原因。

    Returns:
        json.dumps(resume_info): 简历信息的JSON格式。
    """
    resume_info = {
        "name": name,
        "email": email,
        "is_match": is_match,
        "reason": reason,
    }
    return json.dumps(resume_info)


def match_resume(requirements, file_path):
    """
    根据需求匹配简历。

    Args:
        requirements: 需求。
        file_path: 文件路径。

    Returns:
        function_response_data + "\n" + result: 函数响应数据和结果的字符串。
    """
    # 读取简历内容
    file_content = pdf_reader.ocr_pdf_content(file_path)

    # 连接需求和简历内容
    user_query = requirements + "文件内容如下：" + file_content
    print(user_query)

    # 使用OpenAI获取回复
    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[{"role": "user", "content": user_query}],
        functions=function_descriptions,
        function_call="auto",
    )

    ai_response_message = response["choices"][0]["message"]
    ai_response_message_formatted_json = json.dumps(ai_response_message, indent=2, ensure_ascii=False)
    logger.info(ai_response_message_formatted_json)

    # 从OpenAI的响应中提取信息
    name, email, is_match, reason = extract_information_from_response(ai_response_message)

    # 获取简历是否满足要求的信息
    function_response = get_resume_meet_requirements(name=name, email=email, is_match=is_match, reason=reason)

    function_response_json = json.loads(function_response)
    function_response_data = json.dumps(function_response_json, indent=2, ensure_ascii=False)
    logger.info(function_response_data)
    print(function_response_data)

    second_response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {"role": "user", "content": user_query},
            ai_response_message,
            {
                "role": "function",
                "name": "get_resume_meet_requirements",
                "content": function_response,
            },
        ],
    )

    result = second_response['choices'][0]['message']['content']

    return result


# if __name__ == '__main__':
#     req = "我想招聘一位有三年以上工作经验、熟悉Vue3等前端技术栈、期望城市在深圳的前端工程师："
#     file = "/home/lemu-devops/PycharmProjects/resume-pilot/private_upload/2023-06-16-17-41-33/web前端开发工程师 _ 深圳15-20K黄先生 一年以内.pdf"
#     match_resume(req, file)
