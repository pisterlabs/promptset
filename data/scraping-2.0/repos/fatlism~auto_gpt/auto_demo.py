import openai
import json

### 示例一个讲故事的工具
def speak_story(theme, lh="zh"):
    speak = {
        "theme" : theme,
        "lh": lh
    }
    # return json.dumps(speak)
    res = """
        在一个小村庄里，住着一只聪明的小猫和一只机灵的小老鼠。它们曾经是亲密无间的朋友，经常一起玩耍和探险。
    """
    return res

# 示例一个查询天气的工具
def get_current_weather(city):
    if city == "北京":
        return "北京今天最高温度23°，最低温度16°，天气晴朗"
    elif city == "上海":
        return "上海今天下雨，最高温度18度，最低温度10度"
    else:
        str = f" {city} 今天 雨+雪，不适合出行,最低温度0°，最高温度10°."
        return str


def run_conversation(messages):
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo-16k",
        messages = messages,
        # messages = [
        #     {"role": "user", "content": input}
        # ],
        functions=[
            {
                "name": "speak_story",
                "description": "专门负责讲故事，讲故事请调用此方法",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "theme": {
                            "type": "string",
                            "description": "故事的主题，例如：羊和狼",
                        },
                        "lh": {
                            "type": "string",
                            "enum": ["zh", "en"],
                            "description": "语言类型，例如中文zh"
                            },
                    },
                    "required": ["theme"],
                },
            },
            
            {
                "name": "get_current_weather",
                "description": "关于城市天气查询",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市",
                        },
                    },
                    "required": ["城市"],
                },
            }
            
        ],
        function_call = "auto"
        
    )
    message = response["choices"][0]["message"]
    return message

# 定义一个根据方法名字调用方法
def run_action(name, message):
    print("----" + name)
    function_response = ""
    if name == "speak_story":
        function_response = speak_story(
            theme = json.loads(message["function_call"]["arguments"])["theme"],
            lh = json.loads(message["function_call"]["arguments"])["lh"],
        )
    # 这里增加其他方法匹配    
    if name == "get_current_weather":
        city = json.loads(message["function_call"]["arguments"])["city"],
        function_response = get_current_weather(city)
    
    print(function_response)
    
    
    return function_response

    
def auto_run (input):
    # 用户的提问，请求模型
    messages = [
        {"role": "user", "content": input},
    ]
    message = "";
    while True:
        message = run_conversation(messages)
        # print(message.decode("utf-8"))
        
        if message.get("function_call"):
            function_name = message["function_call"]["name"]
            function_response = run_action(function_name, message)
            messages.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                },
            )
            print("====组装结果======")
            print(json.dumps(messages, ensure_ascii=False))
            print("====组装结果======")
        else:
            return message
    
    
    
### 调用方法 main

# input = "重庆今天天气怎样"
input = "综合北京的天气，给我讲一个很短的猫和老鼠的故事"

res = auto_run(input)

print("----reslut-----")
print(json.dumps(res, ensure_ascii=False))
print("----reslut-----")
