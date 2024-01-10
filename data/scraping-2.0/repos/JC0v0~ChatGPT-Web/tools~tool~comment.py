import requests
from config import token
from langchain.tools import tool
@tool
def comment():
    """获取网易云热门评论"""
    url = "https://v2.alapi.cn/api/comment"

    payload = f"token={token}&id="
    headers = {'Content-Type': "application/x-www-form-urlencoded"}

    response = requests.request("POST", url, data=payload, headers=headers)
    data = response.json()["data"]

    return data