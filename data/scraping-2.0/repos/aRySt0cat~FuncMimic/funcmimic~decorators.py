import openai
import json

from funcmimic.prompts import DEFAULT_PROMPT

function_dict = {
    "name": "result_parser",
    "description": "The function to print the result. You always need to call this function.",
    "parameters": {
        "type": "object",
        "properties": {
            "result": {
                "type": "object",
                "description": "the result",
            }
        },
        "required": ["result"],
    },
}

def type_parser(type):
    if type in [int, float]:
        return {"type": "number"}
    if type == str:
        return {"type": "string"}
    if type == bool:
        return {"type": "boolean"}
    if type == list:
        return {"type": "array", "items": {"type": "object"}}
    if type.__name__ == list:
        return {"type": "array", "items": type_parser(type.__args__[0])}


def mimic(prompt=DEFAULT_PROMPT):
    def _mimic(func):
        def wrapper(*args, **kwargs):
            docstring = func.__doc__

            arg_names = func.__code__.co_varnames
            defaults = () if func.__defaults__ is None else func.__defaults__
            args = dict(zip(arg_names, list(args) + list(defaults)))
            args.update(kwargs)
            return_type = func.__annotations__.get("return", None)
            return_type = type_parser(return_type)
            return_type["description"] = "the result"
            function_dict["parameters"]["properties"]["result"] = return_type
            content = prompt.format(docstring=docstring, args=args)
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": content},
                ],
                functions=[function_dict],
                function_call="auto",
            )
            message = response["choices"][0]["message"]
            if "function_call" in message:
                return json.loads(message["function_call"]["arguments"])["result"]
            else:
                print("ChatGPT did not call the function.")
                print("-"*20)
                print(message["content"])
                print("-"*20)
        return wrapper
    return _mimic
