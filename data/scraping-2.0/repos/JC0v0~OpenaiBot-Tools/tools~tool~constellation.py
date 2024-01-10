import requests
from config import token
from langchain.tools import tool
zodiac_dict = {
    '白羊座': 'aries',
    '金牛座': 'taurus',
    '双子座': 'gemini',
    '巨蟹座': 'cancer',
    '狮子座': 'leo',
    '处女座': 'virgo',
    '天秤座': 'libra',
    '天蝎座': 'scorpio',
    '射手座': 'sagittarius',
    '摩羯座': 'capricorn',
    '水瓶座': 'aquarius',
    '双鱼座': 'pisces'
}
@tool
def Horoscope(user_zodiac):
    """输入星座名称，查询该星座运势"""
    if user_zodiac in zodiac_dict:
        zodiac_code = zodiac_dict[user_zodiac]
        url = "https://v2.alapi.cn/api/star"
        payload = f"token={token}&star={zodiac_code}"
        headers = {'Content-Type': "application/x-www-form-urlencoded"}
        response = requests.post(url, data=payload, headers=headers)
        return response.text
    else:
        return "无效的星座"
        


