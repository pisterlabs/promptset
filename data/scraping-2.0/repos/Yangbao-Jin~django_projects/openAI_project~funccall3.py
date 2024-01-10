# 加载环境变量
import openai
import os
import json

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # 读取本地 .env 文件，里面定义了 OPENAI_API_KEY

openai.api_key = os.getenv('OPENAI_API_KEY')



def get_completion(messages, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # 模型输出的随机性，0 表示随机性最小
        function_call="auto",  # 默认值，由系统自动决定，返回function call还是返回文字回复
        functions=[{  # 用 JSON 描述函数。可以定义多个，但是只有一个会被调用，也可能都不会被调用
            "name": "get_location_coordinate",
            "description": "根据POI名称，获得POI的经纬度坐标",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "POI名称，必须是中文",
                    },
                    "city": {
                        "type": "string",
                        "description": "POI所在的城市名，必须是中文",
                    }
                },
                "required": ["location", "city"],
            },
        },
            {
            "name": "search_nearby_pois",
            "description": "搜索给定坐标附近的poi",
            "parameters": {
                "type": "object",
                "properties": {
                    "longitude": {
                        "type": "string",
                        "description": "中心点的经度",
                    },
                    "latitude": {
                        "type": "string",
                        "description": "中心点的纬度",
                    },
                    "keyword": {
                        "type": "string",
                        "description": "目标poi的关键字",
                    }
                },
                "required": ["longitude", "latitude", "keyword"],
            },
        }],
    )
    return response.choices[0].message

import requests

amap_key = "6d672e6194caa3b639fccf2caf06c342"


def get_location_coordinate(location, city="北京"):
    url = f"https://restapi.amap.com/v5/place/text?key={amap_key}&keywords={location}&region={city}"
    print(url)
    r = requests.get(url)
    result = r.json()
    if "pois" in result and result["pois"]:
        return result["pois"][0]
    return None


def search_nearby_pois(longitude, latitude, keyword):
    url = f"https://restapi.amap.com/v5/place/around?key={amap_key}&keywords={keyword}&location={longitude},{latitude}"
    print(url)
    r = requests.get(url)
    result = r.json()
    ans = ""
    if "pois" in result and result["pois"]:
        for i in range(min(3, len(result["pois"]))):
            name = result["pois"][i]["name"]
            address = result["pois"][i]["address"]
            distance = result["pois"][i]["distance"]
            ans += f"{name}\n{address}\n距离：{distance}米\n\n"
    return ans

prompt = "沈阳市太原街附近的咖啡"

messages = [
    {"role": "system", "content": "你是一个地图通，你可以找到任何地址。"},
    {"role": "user", "content": prompt}
]
response = get_completion(messages)
messages.append(response)  # 把大模型的回复加入到对话中
print("=====GPT回复=====")
print(response)

# 如果返回的是函数调用结果，则打印出来
while (response.get("function_call")):
    if (response["function_call"]["name"] == "get_location_coordinate"):
        args = json.loads(response["function_call"]["arguments"])
        print("Call: get_location_coordinate")
        result = get_location_coordinate(**args)
        print(result)
    elif (response["function_call"]["name"] == "search_nearby_pois"):
        args = json.loads(response["function_call"]["arguments"])
        print("Call: search_nearby_pois")
        result = search_nearby_pois(**args)
        #print("=====函数返回=====")
        print(result)
    messages.append(
        {"role": "function", "name": response["function_call"]["name"], "content": str(
            result)}  # 数值result 必须转成字符串
    )
    response = get_completion(messages)

print("=====最终回复=====")
print(get_completion(messages).content)
print()

print(messages)
