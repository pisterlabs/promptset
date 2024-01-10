import requests
from config import token
from langchain.tools import tool
@tool
def get_weather(city):
    """输入城市名称获取相应城市天气预报"""
    url = "https://v2.alapi.cn/api/tianqi"

    # 编码城市名为 utf-8 格式
    encoded_city = city.encode('utf-8')
    
    payload = {
        'token': token,
        'city': encoded_city.decode('utf-8')
    }

    response = requests.post(url, data=payload)

    return response.text

