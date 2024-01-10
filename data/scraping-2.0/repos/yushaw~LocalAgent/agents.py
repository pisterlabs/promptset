import json
import openai
from typing import Optional
from tool import toolParser as parser

# function parser 部分的代码来自于 https://github.com/aurelio-labs/funkagent

'''
    1. 整体流程
    用户提出一个问题，调用 ask 方法。

    1/ 在 ask 方法中，首先重置internal_thoughts（内部思考），这是一个列表，用来保存在处理问题时的一些中间步骤或结果。
    2/ 然后将用户的问题添加到 chat_history（聊天历史）列表中。
    3/ 然后调用 _generate_response 方法生成响应。在这个方法中，一个无限循环开始。每次循环，都会调用 _create_chat_completion 方法来创建一个ChatCompletion对象，
    并尝试从GPT-4模型中获取一个响应。如果这个响应的完成原因是'stop'，或者internal_thoughts的长度超过了3，那么循环结束。否则，如果完成原因是'function_call'，那么就调用 _handle_function_call 方法来处理函数调用。
    4/ 在 _handle_function_call 方法中，首先将函数调用的信息添加到 internal_thoughts 中。然后从函数调用信息中提取函数名和参数，并调用 _call_function 方法来执行函数。执行完函数后，将函数的结果添加到 internal_thoughts 中。
    5/ 无限循环结束后，调用 _final_thought_answer 方法创建最终的思考，包括所有的中间思考步骤。然后再次调用 _create_chat_completion 方法来创建一个ChatCompletion对象，这次不再考虑函数调用。
    最后，将这个最终的响应添加到 chat_history 中，并返回这个响应。
    
    2. Function calling 的 api 链接地址
    https://openai.com/blog/function-calling-and-other-api-updates

'''

sys_msg = """
You are an assistant designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussion on a wide range of topics. 
    As a language model, Assistant is able to generate human-like text based on the input it receives, 
    allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand. \n
Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, 
and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, 
allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.\n
Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. 
Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
"""

class Agent:
    def __init__(
        self,
        openai_api_key: str,
        model_name: str = "gpt-3.5-turbo",
        functions: Optional[list] = None,
        thinkDepth: int = 3,
    ):
        openai.api_key = openai_api_key
        self.model_name = model_name
        self.functions = self._parse_functions(functions)
        self.func_mapping = self._create_func_mapping(functions)
        self.chat_history = [{'role': 'system', 'content': sys_msg}]
        self.thinkDepth = thinkDepth

    def _parse_functions(self, functions: Optional[list]) -> Optional[list]:
        '''
            解析输入的函数列表，返回每个函数的json格式。
        '''
        if functions is None:
            return None
        return [parser.func_to_json(func) for func in functions]

    def _create_func_mapping(self, functions: Optional[list]) -> dict:
        '''
            创建一个函数名到函数的映射。
        '''
        if functions is None:
            return {}
        return {func.__name__: func for func in functions}

    def _create_chat_completion(
        self, messages: list, use_functions: bool=True
    ) -> openai.ChatCompletion:
        '''
            创建一个chat completion对象。如果提供了函数并设置使用函数，则将函数名加入到创建的对象中。
        '''
        # 如果使用函数而且函数列表不为空，那么就把函数的名字加进去
        if use_functions and self.functions:
            res = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                functions=self.functions
            )
        else:
            res = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages
            )
        return res

    def _generate_response(self) -> openai.ChatCompletion:
        '''
            生成响应。这是一个无限循环，直到返回的消息的完成原因是'stop'或内部思考超过3次。然后创建最终的回答并返回。
        '''
        while True:
            print('.', end='')
            res = self._create_chat_completion(
                self.chat_history + self.internal_thoughts
            )
            finish_reason = res.choices[0].finish_reason

            if finish_reason == 'stop' or len(self.internal_thoughts) > self.thinkDepth:
                # create the final answer
                final_thought = self._final_thought_answer()
                final_res = self._create_chat_completion(
                    self.chat_history + [final_thought],
                    use_functions=False
                )
                return final_res
            elif finish_reason == 'function_call':
                self._handle_function_call(res)
            else:
                raise ValueError(f"Unexpected finish reason: {finish_reason}")

    def _handle_function_call(self, res: openai.ChatCompletion):
        '''
            如果返回的消息的完成原因是'function_call'，处理函数调用。
        '''
        self.internal_thoughts.append(res.choices[0].message.to_dict())
        func_name = res.choices[0].message.function_call.name
        args_str = res.choices[0].message.function_call.arguments
        result = self._call_function(func_name, args_str)
        print(result)
        res_msg = {'role': 'assistant', 'content': (f"The answer is {result}.")}
        self.internal_thoughts.append(res_msg)

    def _call_function(self, func_name: str, args_str: str):
        '''
            调用函数并返回结果。
        '''
        args = json.loads(args_str)
        func = self.func_mapping[func_name]
        res = func(**args)
        return res
    
    def _final_thought_answer(self):
        '''
            创建最终的思考，包括前面所有的思考步骤。
        '''
        thoughts = ("To answer the question I will use these step by step instructions."
                    "\n\n")
        for thought in self.internal_thoughts:
            if 'function_call' in thought.keys():
                thoughts += (f"I will use the {thought['function_call']['name']} "
                             "function to calculate the answer with arguments "
                             + thought['function_call']['arguments'] + ".\n\n")
            else:
                thoughts += thought["content"] + "\n\n"
        self.final_thought = {
            'role': 'assistant',
            'content': (f"{thoughts} Based on the above, I will now answer the "
                        "question, this message will only be seen by me so answer with "
                        "the assumption with that the user has not seen this message.")
        }
        return self.final_thought

    def ask(self, query: str) -> openai.ChatCompletion:
        '''
            用户提出问题后的主要流程。首先，重置内部思考，然后将问题添加到聊天历史中，生成响应并将响应添加到聊天历史中，最后返回响应。
        '''
        self.internal_thoughts = []
        self.chat_history.append({'role': 'user', 'content': query})
        res = self._generate_response()
        self.chat_history.append(res.choices[0].message.to_dict())
        return res