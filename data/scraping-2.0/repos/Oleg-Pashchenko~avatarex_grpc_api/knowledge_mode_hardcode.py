import json

import openai

descr = "Ищет соотвтествующий вопрос если не нашел соотвтествия - возвращает пустоту"


async def get_keywords_values(message, func, openai_api_key):
    async with openai.AsyncOpenAI(api_key=openai_api_key) as client:
        try:
            messages = [
                {'role': 'system', 'content': descr},
                {"role": "user",
                 "content": message}]

            response = await client.chat.completions.create(model="gpt-3.5-turbo-16k",
                                                            messages=messages,
                                                            functions=func,
                                                            function_call={"name": "get_question_by_context"})
            response_message = response.choices[0].message
        except Exception as e:
            print("ERROR", e)
            return {'is_ok': False, 'args': {}}
        if response_message.function_call:
            function_args = json.loads(response_message.function_call.arguments)
            print(function_args.keys())
            try:
                if len(list(function_args.keys())) == 0:
                    return {'is_ok': False, 'args': []}
                return {'is_ok': True, 'args': list(function_args.keys())}
            except:
                return {'is_ok': False, 'args': []}
        else:
            return {'is_ok': False, 'args': []}


def get_function(data):
    properties = {}
    rq = []
    for r in data:
        if r in rq:
            continue
        rq.append(r[0])
        properties[r[0]] = {'type': 'boolean', 'description': 'Вопрос полностью соответствует заданному?'}

    return [{
        "name": "get_question_by_context",
        "description": descr,
        "parameters": {
            "type": "object",
            "properties": properties,
            'required': rq
        }
    }]


def get_answer(q, data):
    for d in data:
        if d[0] == q:
            return d[1]
    return "Ошибка поиска в базе знаний!"


async def main(message, data, openai_key):
    response = await get_keywords_values(message, get_function(data), openai_key)
    if not response['is_ok']:
        return False, ''
    else:
        return True, get_answer(response['args'][0], data)
