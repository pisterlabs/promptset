import requests
from config import token
from langchain.tools import tool

@tool
def kd(tool_input):
    """根据快递单号查询快递信息"""
    url = "https://v2.alapi.cn/api/kd"

    payload = f"token={token}&number={tool_input}"
    headers = {'Content-Type': "application/x-www-form-urlencoded"}

    response = requests.post(url, data=payload, headers=headers)

    return response.text
