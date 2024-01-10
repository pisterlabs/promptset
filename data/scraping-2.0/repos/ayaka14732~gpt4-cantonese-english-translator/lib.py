import json
import openai

def must_return_a_string(response: str) -> str:
    assert isinstance(response, str)
    return json.dumps({
        'response': response,
    })

INITIAL_PROMPT = '''You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.
Knowledge cutoff: 2021-09
Current date: 2023-08-16'''

DOCSTRING = """
Serializes the given `response` string to a JSON format.

Parameters:
- response (str): The response string to be serialized.

Returns:
- str: A JSON string containing the provided response.

Raises:
- AssertionError: If the provided `response` is not a string.

Examples:
>>> must_return_a_string("hello")
'{"response": "hello"}'

Note:
This function makes use of assertions to ensure data consistency. It's essential to provide the 
required response as a string before calling this function.
"""

must_return_a_string.__doc__ = DOCSTRING

def yue2en(input_string: str) -> str:
    messages = [
        {'role': 'system', 'content': INITIAL_PROMPT},
        {'role': 'user', 'content': '我而家做緊一本廣東話辭典，要寫好多例句。我想請你幫我翻譯一啲句子，但一定要譯得高質，得唔得？如果句子有錯，你都要照譯，唔好輸出其他嘢。'},
        {'role': 'assistant', 'content': 'OK，梗係冇問題啦！你唔知我係廣東話專家咩？'},
        {'role': 'user', 'content': '佢咁勁，實做到啦。'},
        {'role': 'assistant', 'content': 'She is talented. She certainly can do it.'},
        {'role': 'user', 'content': '啲人走晒之後，呢度終於靜返。'},
        {'role': 'assistant', 'content': 'After everyone had left, this place became quiet again.'},
        {'role': 'user', 'content': '畢業之後，佢好耐冇蒲頭，但我間唔中聽人講起佢做緊深度學習嘅嘢。'},
        {'role': 'assistant', 'content': 'After graduation, he disappeared for a long time, but I occasionally hear about him doing things related to deep learning.'},
        {'role': 'user', 'content': 'SoMeGarBAGEStriNG!!###'},
        {'role': 'assistant', 'content': 'SoMeGarBAGEStriNG!!###'},
        {'role': 'user', 'content': input_string},
    ]
    functions = [
        {
            'name': 'must_return_a_string',
            'description': DOCSTRING,
            'parameters': {
                'type': 'object',
                'properties': {
                    'response': {
                        'type': 'string',
                        'description': "The 'response' string in the given object",
                    },
                },
                'required': ['response'],
            },
        }
    ]
    response = openai.ChatCompletion.create(
        model='gpt-4-0613',
        messages=messages,
        temperature=0,
        functions=functions,
        function_call={'name': 'must_return_a_string'},
    )
    response_message = response['choices'][0]['message']
    function_call = response_message['function_call']
    assert function_call['name'] == 'must_return_a_string'
    result = json.loads(function_call['arguments'])['response']
    return result

def en2yue(input_string: str) -> str:
    messages = [
        {'role': 'system', 'content': INITIAL_PROMPT},
        {'role': 'user', 'content': '我而家做緊一本廣東話辭典，要寫好多例句。我想請你幫我翻譯一啲句子，但一定要譯得高質，得唔得？如果句子有錯，你都要照譯，唔好輸出其他嘢。'},
        {'role': 'assistant', 'content': 'OK，梗係冇問題啦！你唔知我係廣東話專家咩？'},
        {'role': 'user', 'content': 'She is talented. She certainly can do it.'},
        {'role': 'assistant', 'content': '佢咁勁，實做到啦。'},
        {'role': 'user', 'content': 'After everyone had left, this place became quiet again.'},
        {'role': 'assistant', 'content': '啲人走晒之後，呢度終於靜返。'},
        {'role': 'user', 'content': 'After graduation, he disappeared for a long time, but I occasionally hear about him doing things related to deep learning.'},
        {'role': 'assistant', 'content': '畢業之後，佢好耐冇蒲頭，但我間唔中聽人講起佢做緊深度學習嘅嘢。'},
        {'role': 'user', 'content': 'SoMeGarBAGEStriNG!!###'},
        {'role': 'assistant', 'content': 'SoMeGarBAGEStriNG!!###'},
        {'role': 'user', 'content': input_string},
    ]
    functions = [
        {
            'name': 'must_return_a_string',
            'description': DOCSTRING,
            'parameters': {
                'type': 'object',
                'properties': {
                    'response': {
                        'type': 'string',
                        'description': "The 'response' string in the given object",
                    },
                },
                'required': ['response'],
            },
        }
    ]
    response = openai.ChatCompletion.create(
        model='gpt-4-0613',
        messages=messages,
        temperature=0,
        functions=functions,
        function_call={'name': 'must_return_a_string'},
    )
    response_message = response['choices'][0]['message']
    function_call = response_message['function_call']
    assert function_call['name'] == 'must_return_a_string'
    result = json.loads(function_call['arguments'])['response']
    return result
