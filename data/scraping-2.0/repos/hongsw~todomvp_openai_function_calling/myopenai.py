
import openai
import os
import main3
import json
from TodoModel import Todo

openai.api_key = os.getenv("OPENAI_API_KEY")

class MyOpenAI:
    def get_current_weather(location, unit="섭씨"):
        weather_info = {
            "location": location,
            "temperature": "24",
            "unit": unit,
            "forecast": ["sunny", "windy"],
        }
        return weather_info

    async def chat(ask: str = "지금 서울날씨를 섭씨로 알려줘.", output: str = "json"):
        print("ask : ", ask)
        messages = [{"role": "user", "content": ask}]
        # openapi_schema = main3.get_openapi()
        # print(openapi_schema);
        print(Todo.schema())
        functions = [
            {
                "name": "get_current_weather",
                "description": "특정 지역의 날씨를 알려줍니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "지역이름 eg. 서울, 부산, 제주도",
                        },
                        "unit": {"type": "string", "enum": ["섭씨", "화씨"]},
                    },
                    "required": ["location"],
                },
            },
            {
                "name": "_get_all_routes",
                "description": "현재 서버에 등록된 API 목록을 보여줍니다.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "POST_todos__todo",
                "description": "현재 서버에 등록된 API 목록을 보여줍니다.",
                "parameters": {'title': 'Todo', 'type': 'object', 'properties':  {'title': {'title': 'Title', 'type': 'string'}, 'completed': {'title': 'Completed', 'default': False, 'type': 'boolean'}}, 'required': ['title']}
            },
            {
                "name": "PUT_todos__id",
                "description": "현재 서버에 등록된 API 목록을 보여줍니다.",
                "parameters": {
                                "type": "object",
                                    "properties": {
                                        "id": {
                                        "title": "Id",
                                        "type": "integer"
                                        }
                                    },
                }
            },
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
            functions=functions,
            function_call="auto",
            )
        response_message = response["choices"][0]["message"]

        print("response_message1 : ", response_message)

        if response_message.get("function_call"):
            # Note: the JSON response may not always be valid; be sure to handle errors
            available_functions = {
                "_get_all_routes": main3._get_all_routes,
                "POST_todos__todo": main3.create_todo,
                "PUT_todos__id": main3.update_todo,
                "get_current_weather": MyOpenAI.get_current_weather,
            }
            function_name = response_message["function_call"]["name"]
            fuction_to_call = available_functions[function_name]
            function_args = json.loads(response_message["function_call"]["arguments"])
            function_response = ''
            print('function_args.get("title") :')
            print(function_args.get("title"))
            if("get_current_weather" == function_name):
                function_response = fuction_to_call(
                    location=function_args.get("location"),
                    unit=function_args.get("unit"),
                )
            elif("POST_todos__todo" == function_name):
                print('POST_todos__todo')
                function_response = await fuction_to_call(
                    Todo( title=function_args.get("title"))
                )
                function_response = function_response.to_dict()
                function_response = 'done POST_todos__todo'
            elif("PUT_todos__id" == function_name):
                 function_response = fuction_to_call(
                    id=function_args.get("Id")
                )
            elif("_get_all_routes" == function_name):
                function_response = fuction_to_call()
            else:
                raise Exception("Unknown function name")

            if(output == "json"):
                return function_response
            
            print("\n second : ", messages)
            print('function_response :')
            print(function_response)
            messages.append(response_message)
            messages.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                }
            )
            
            print("\n second : ", messages)
            
            second_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=messages,
            )  # get a new response from GPT where it can see the function response

            json_data = json.dumps(second_response, ensure_ascii=False)

            print("second_response : ", second_response)
            print("second_response message : ", second_response.choices[0].message.content)
            return second_response.choices[0].message.content
