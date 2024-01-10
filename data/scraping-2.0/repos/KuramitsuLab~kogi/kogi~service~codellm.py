import os
import openai
from .textra import EJ
from .s3logging import record_log

def AUTH_ERROR():
    return EJ("""
Please set your own API_KEY.
                       
```python
import os
os.environ["OPENAI_API_KEY"] = PLEASE SET YOUR OWN API KEY.
```
                       
""", """
ご自分のAPIキーを設定してください。

```
import os
os.environ["OPENAI_API_KEY"] = 自分のAPIキーを設定してください.
```

""")

def COMM_ERROR():
    return EJ("""
We're experiencing some connectivity issues right now. 
Taking a brief pause might help things return to normal.

""", """
通信エラーが発生しています。
少し時間を空けると、正常に戻ることがあります。

""")

def TOO_MANY_USAGES():
    return EJ("""
You seem to be relying quite a bit on AI. 
How about taking a break for today?

""", """
あなたは、AIに頼りすぎています。
今日はこのくらいにしてみてはいかがでしょうか？

""")

_TEMPORARY=None
llm_count = 0
model_cache = {}

def _prepare_messages(prompt, context):
    messages = context.get('messages', [])
    messages.append({
        'role': 'user', 'content': prompt
    })
    context['messages'] = messages
    return messages[:]

def _messages_to_text(messages):
    ss = []
    for m in messages:
        if 'role' in m:
            ss.append(f"{m['role'].capitalize()}:")
        if 'content' in m:
            ss.append(f"{m['content']}\n")
    return '\n'.join(ss)

def llm_prompt(prompt, context: dict):
    global model_cache, llm_count
    messages = _prepare_messages(prompt, context)
    key = '|'.join(d['content'] for d in messages)
    if key in model_cache:
        return model_cache[key]

    if llm_count > 128:
        return TOO_MANY_USAGES()
    llm_count+=1

    if 'prompt_suffix' in context:
        messages.append({'role': 'user', 'content': context['prompt_suffix']})
    if '_prompt_suffix' in context:
        messages.append({'role': 'user', 'content': context['_prompt_suffix']})
    try:
        openai.api_key = os.environ.get('OPENAI_API_KEY', _TEMPORARY)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        openai.api_key = None
        model_name = response["model"]
        used_tokens = response["usage"]["total_tokens"]
        response = response.choices[0]["message"]["content"].strip()
        record_log(
            log='llm', model=model_name,
            prompt = _messages_to_text(messages),
            response = response,
            messages = messages,
        )
        context['messages'].append({'role': 'assistant', 'content': response})
        context['tokens'] = context.get('tokens', 0) + used_tokens
        model_cache[key] = response
        return {
            'content': response,
            'tokens': used_tokens
        }
    except openai.error.AuthenticationError as e:
        return {
            'whoami': '@system',
            'content': AUTH_ERROR(),
        }
    except BaseException as e: #openai.error.ServiceUnavailableError as e:
        return {
            'whoami': '@system',
            'content': COMM_ERROR(),
        }

def llm_login(apikey):
    global _TEMPORARY, llm_count
    if llm_count > 128:
        return TOO_MANY_USAGES()
    llm_count+=1
    try:
        openai.api_key = apikey
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {'role': 'user', 'content': 'Are you ready?'}
            ],
        )
        openai.api_key = None
        # used_tokens = response["usage"]["total_tokens"]
        response = response.choices[0]["message"]["content"].strip()
        _TEMPORARY = apikey
        return True
    except openai.error.AuthenticationError as e:
        return False
    except BaseException as e: 
        return False
