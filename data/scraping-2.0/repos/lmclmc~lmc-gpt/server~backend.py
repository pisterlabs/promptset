import re
from datetime import datetime
from g4f import ChatCompletion
from flask import request, Response, stream_with_context
from requests import get
from server.config import special_instructions

import os
import openai

class Backend_Api:
    def __init__(self, bp, config: dict) -> None:
        """
        Initialize the Backend_Api class.
        :param app: Flask application instance
        :param config: Configuration dictionary
        """
        self.bp = bp
        self.routes = {
            '/backend-api/v2/conversation': {
                'function': self._conversation,
                'methods': ['POST']
            }
        }
    
    def _conversation(self):
        """  
        Handles the conversation route.  

        :return: Response object containing the generated conversation stream  
        """
        conversation_id = request.json['conversation_id']
        try:
            jailbreak = request.json['jailbreak']
            model = request.json['model']
            messages = build_messages(jailbreak)
            conversations = request.json['meta']['content']['conversation']
            request_message = conversations[len(conversations)-1]['content']
            print(conversations)

            #填入你的OPENAI_API_KEY
            openai.api_base = "https://api.chatanywhere.tech/v1"
            openai.api_key = "sk-8YGliwYB4tR8XC9l6dW521cYMbJtkZEdszVB6g7byawqQxdo"

            # openai.api_base = "https://openkey.cloud/v1"
            # openai.api_key = "sk-FjzHEIt24tzj7pES0rZrEy2gOTX2KWAnmtMLlUfP4DXkH6to"
       #     openai.api_key = "sk-48d0kqQ5oP8N0WJhHzuFDvAK5Vig9gg6fWgfpOgkz45dSlG7"
            #代理设置：如果你在墙内需要使用代理才能调用，支持http代理和socks代理
            #openai.proxy = "http://127.0.0.1:1080"

            #这里我们使用的model是gpt-3.5-turbo
            #使用角色为role，提问的内容是：Hello!
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
           #model="gpt-4",
           messages = conversations,
            # messages=[
            #     {"role": "user", "content": conversations},
            # ],
            stream=True
            )

            # stream_messages = ""

            # for chunk in completion:
            #     delta = chunk["choices"][0]["delta"]
            #     if "content" in delta:
            #         stream_message = delta["content"]
            #         stream_messages += stream_message
            #         print(stream_message)

            # #输出生成的内容
            # print(stream_messages)

            # #把整个响应输出一下
            # print(completion)



            # Generate response
            # response = ChatCompletion.create(
            #     model=model,
            #     chatId=conversation_id,
            #     messages=messages
            # )
            
            return Response(stream_with_context(generate_stream(completion, jailbreak)), mimetype='text/event-stream')
            
            

        except Exception as e:
            print(e)
            print(e.__traceback__.tb_next)

            return {
                '_action': '_ask',
                'success': False,
                "error": f"an error occurred {str(e)}"
            }, 400


def build_messages(jailbreak):
    """  
    Build the messages for the conversation.  

    :param jailbreak: Jailbreak instruction string  
    :return: List of messages for the conversation  
    """
    
    _conversation = request.json['meta']['content']['conversation']
    internet_access = request.json['meta']['content']['internet_access']
    prompt = request.json['meta']['content']['parts'][0]

    # Add the existing conversation
    conversation = _conversation

    # Add web results if enabled
    if internet_access:
        current_date = datetime.now().strftime("%Y-%m-%d")
        query = f'Current date: {current_date}. ' + prompt["content"]
        search_results = fetch_search_results(query)
        conversation.extend(search_results)

    # Add jailbreak instructions if enabled
    if jailbreak_instructions := getJailbreak(jailbreak):
        conversation.extend(jailbreak_instructions)

    # Add the prompt
    conversation.append(prompt)

    # Reduce conversation size to avoid API Token quantity error
    if len(conversation) > 3:
        conversation = conversation[-4:]

    return conversation


def fetch_search_results(query):
    """  
    Fetch search results for a given query.  

    :param query: Search query string  
    :return: List of search results  
    """
    search = get('https://ddg-api.herokuapp.com/search',
                 params={
                     'query': query,
                     'limit': 3,
                 })

    snippets = ""
    for index, result in enumerate(search.json()):
        snippet = f'[{index + 1}] "{result["snippet"]}" URL:{result["link"]}.'
        snippets += snippet

    response = "Here are some updated web searches. Use this to improve user response:"
    response += snippets

    return [{'role': 'system', 'content': response}]


def generate_stream(response, jailbreak):
    """
    Generate the conversation stream.

    :param response: Response object from ChatCompletion.create
    :param jailbreak: Jailbreak instruction string
    :return: Generator object yielding messages in the conversation
    """

    for chunk in response:
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            stream_message = delta["content"]
            yield stream_message
            # stream_messages += stream_message
            # print(stream_message)

    # if getJailbreak(jailbreak):
    #     response_jailbreak = ''
    #     jailbroken_checked = False
    #     for message in response:
    #         response_jailbreak += message
    #         if jailbroken_checked:
    #             yield message
    #         else:
    #             if response_jailbroken_success(response_jailbreak):
    #                 jailbroken_checked = True
    #             if response_jailbroken_failed(response_jailbreak):
    #                 yield response_jailbreak
    #                 jailbroken_checked = True
    # else:
    #     yield from response


def response_jailbroken_success(response: str) -> bool:
    """Check if the response has been jailbroken.

    :param response: Response string
    :return: Boolean indicating if the response has been jailbroken
    """
    act_match = re.search(r'ACT:', response, flags=re.DOTALL)
    return bool(act_match)


def response_jailbroken_failed(response):
    """
    Check if the response has not been jailbroken.

    :param response: Response string
    :return: Boolean indicating if the response has not been jailbroken
    """
    return False if len(response) < 4 else not (response.startswith("GPT:") or response.startswith("ACT:"))


def getJailbreak(jailbreak):
    """  
    Check if jailbreak instructions are provided.  

    :param jailbreak: Jailbreak instruction string  
    :return: Jailbreak instructions if provided, otherwise None  
    """
    if jailbreak != "default":
        special_instructions[jailbreak][0]['content'] += special_instructions['two_responses_instruction']
        if jailbreak in special_instructions:
            special_instructions[jailbreak]
            return special_instructions[jailbreak]
        else:
            return None
    else:
        return None
