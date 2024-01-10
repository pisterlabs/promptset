import openai
import json

# openai.api_key = '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'
from api_keys import openai_api_key # API key가 github에 올라가면 폐기되기 때문에 따로 import 했습니다.
openai.api_key=openai_api_key  # API key가 github에 올라가면 폐기되기 때문에 따로 import 했습니다.

# get_current_weather 함수를 실제로 구현한다면 실제 날씨 정보 API를 이용해야 하지만,
# 여기서는 예시를 위해 간단하게 하드 코딩된 함수를 제공합니다.
def get_current_weather(location, unit="fahrenheit"):
    """location으로 받은 지역의 날씨를 알려 주는 기능"""
    weather_info={
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)

def run_conversation():
    # 1단계: messages뿐만 아니라 사용할 수 있는 함수에 대한 설명 추가하기
    messages=[{"role": "user", "content": "What's the weather like in Boston?"}]
    functions=[
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
            "required": ["location"],
            },
        }
    ]
    
    response=openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto", # auto가 기본 설정입니다.
    )
    response_message=response["choices"][0]["message"]
 
    # 2단계: GPT의 응답이 function을 실행해야 한다고 판단했는지 확인하기
    if response_message.get("function_call"):
        # 3단계: 해당 함수 실행하기
        available_functions={
            "get_current_weather": get_current_weather,
        } # 이 예제에서는 사용할 수 있는 함수가 하나뿐이지만, 여러 개를 설정할 수 있습니다.
        function_name=response_message["function_call"]["name"]
        fuction_to_call=available_functions[function_name]
        function_args=json.loads(response_message["function_call"]["arguments"])
        function_response=fuction_to_call(
            location=function_args.get("location"),
            unit=function_args.get("unit"),
        )
        # 4단계: 함수를 실행한 결과를 GPT에게 보내 답을 받아오기 위한 부분
        messages.append(response_message) # GPT의 지난 답변을 messages에 추가하기
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        ) # 함수 실행 결과도 GPT messages에 추가하기
        second_response=openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
        ) # 함수 실행 결과를 GPT에 보내 새로운 답변 받아오기
        return second_response
    
print(run_conversation())
