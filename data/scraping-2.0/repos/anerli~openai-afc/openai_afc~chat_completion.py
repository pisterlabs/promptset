import json
from typing import List
import openai
from openai_afc import AutoFnDefinition
from openai_afc.errors import GPTSchemaDeviation

class AutoFnChatCompletion(openai.ChatCompletion):
    @classmethod
    def create(cls, functions: List[AutoFnDefinition] = None, *args, **kwargs):
        """
        functions: list of function definitions to be auto-called when deemed necessary by GPT.

        Modified version of create chat completion API, which takes Python functions
        and automatically calls them if GPT requests to do so.

        Note that this "intercepts" the `functions` argument, so it must be specified differently
        than described in the OpenAI documentation.

        If n != 1, then intermediate function calls may be sending several wasted responses,
        since only the first choice is used when a function is used.
        """
        messages = kwargs['messages'][:]
        del kwargs['messages']
        fn_map = {}
        fn_metadata = []
        for fn in functions:
            fn_map[fn.name] = fn
            fn_metadata.append(fn.get_metadata())
        
        while True:
            resp = super().create(*args, **kwargs, messages=messages, functions=fn_metadata)

            msg = resp['choices'][0]['message']

            if msg['content']:
                # Keep interface same as ChatCompletion
                return resp
            else:
                name = msg['function_call']['name']
                try:
                    gpt_kw_arguments = json.loads(msg['function_call']['arguments'])
                except json.decoder.JSONDecodeError:
                    raise GPTSchemaDeviation('GPT gave non-JSON function call arguments:', msg['function_call']['arguments'])
                result = fn_map[name].fn(**gpt_kw_arguments)

                messages.append({
                    "role": "function",
                    "name": name,
                    "content": fn_map[name].output_transform(result),
                })