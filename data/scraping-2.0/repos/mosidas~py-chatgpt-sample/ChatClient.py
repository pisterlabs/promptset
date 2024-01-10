import json
import openai
import function_calling
import time

class ChatClient:
    def __init__(self, api_key, model, functions, system_prompt=None):
        """
        チャットクライアントを初期化する
        """
        self.api_key = api_key
        self.past_messages = []
        self.model = model
        self.functions = functions
        if system_prompt is not None:
            self.past_messages.append({"role": "system", "content": system_prompt})
        openai.api_key = self.api_key

    def chat(self, message):
        """
        メッセージを送信し、AIからの返答とトークン数を返す
        """
        self.past_messages.append({"role": "user", "content": message})

        # send prompt
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.past_messages,
            #temperature=1,
            #top_p=1,
        )

        answer = response.choices[0].message["content"]
        tokens = response.usage["total_tokens"]
        self.past_messages.append({"role": "assistant", "content": answer})

        return answer, tokens

    def chat_stream(self, message):
        """
        メッセージを送信し、AIからの返答とトークン数を返す
        メッセージはストリームで返される、空文字列が返されたら終了
        トークン数は常に0を返す。(responseにtotal_tokensが含まれないため)
        """
        self.past_messages.append({"role": "user", "content": message})

        # send prompt
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.past_messages,
            temperature=0.5,
            #top_p=1,
            stream=True,
        )

        chunked_message = []
        for chunk in response:
            if("content" in chunk['choices'][0]['delta']):
                message = chunk['choices'][0]['delta'].get('content')
                chunked_message.append(message)
                yield message, 0
            else:
                pass

        self.past_messages.append({"role": "assistant", "content": "".join(chunked_message)})
        yield "", 0

    def chat_with_function_call(self, message):
        """
        メッセージを送信し、AIからの返答とトークン数を返す
        function_callを使用する
        """
        self.past_messages.append({"role": "user", "content": message})

        # send prompt
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.past_messages,
            functions=self.functions,
            function_call="auto",
            #temperature=1,
            #top_p=1,
        )

        message = response["choices"][0]["message"]

        if message.get("function_call"):
            func_name = message["function_call"]["name"]
            arguments = json.loads(message["function_call"]["arguments"])
            func = getattr(function_calling, func_name)
            function_response = func(arguments.get("item"))

            self.past_messages.append({
                "role": "function",
                "name": func_name,
                "content": function_response
                })

            second_answer = openai.ChatCompletion.create(
                model=self.model,
                messages=self.past_messages,
            )

            answer = second_answer.choices[0].message["content"]
            total_tokens = second_answer.usage["total_tokens"]

            return answer, total_tokens
        else:
            answer = response.choices[0].message["content"]
            total_tokens = response.usage["total_tokens"]

            self.past_messages.append({"role": "assistant", "content": answer})

            return answer, total_tokens

