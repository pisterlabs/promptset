import json
import openai
import function_calling

class ChatClientAzure:
    def __init__(self, api_key, api_base, engine, api_version, system_prompt=None):
        """
        チャットクライアントを初期化する
        """
        self.api_key = api_key
        self.past_messages = []
        self.engine = engine
        self.system_prompt = system_prompt
        if system_prompt is not None:
            self.past_messages.append({"role": "system", "content": system_prompt})
        openai.api_key = api_key
        openai.api_type = "azure"
        openai.api_base = api_base
        openai.api_version = api_version

    def reset(self):
        """
        メッセージ履歴をリセットする
        """
        self.past_messages = []
        if self.system_prompt is not None:
            self.past_messages.append({"role": "system", "content": self.system_prompt})


    def chat(self, message):
        """
        メッセージを送信し、AIからの返答とトークン数を返す
        """
        self.past_messages.append({"role": "user", "content": message})

        # send prompt
        response = openai.ChatCompletion.create(
            engine=self.engine,
            messages=self.past_messages,
            temperature=0.5,
            #top_p=1,
        )

        answer = response.choices[0].message["content"]
        tokens = response.usage["total_tokens"]
        self.past_messages.append({"role": "assistant", "content": answer})

        return answer, tokens

    def chat_on_your_data(self, message, data_sources):
        self.past_messages.append({"role": "user", "content": message})

        # send prompt
        response = openai.ChatCompletion.create(
            engine=self.engine,
            messages=self.past_messages,
            temperature=0.5,
            #top_p=1,
            dataSources=data_sources
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
            engine=self.engine,
            messages=self.past_messages,
            temperature=0.5,
            #top_p=1,
            stream=True,
        )

        chunked_message = []
        for chunk in response:
            if(len(chunk['choices']) == 0):
                pass
            else:
                message = chunk['choices'][0]['delta'].get('content')
                if(message == None):
                    pass
                else:
                    chunked_message.append(message)
                    yield message, 0

        self.past_messages.append({"role": "assistant", "content": "".join(chunked_message)})
        yield "", 0

    def chat_with_function_call(self, message, functions):
        """
        メッセージを送信し、AIからの返答とトークン数を返す
        function_callを使用する
        """
        self.past_messages.append({"role": "user", "content": message})

        # send prompt
        response = openai.ChatCompletion.create(
            engine=self.engine,
            messages=self.past_messages,
            functions=functions,
            function_call="auto",
            temperature=1
        )

        message = response["choices"][0]["message"]

        # AIがfunction_calliingが必要と判断した場合、レスポンスにfunction_callが含まれる
        if message.get("function_call"):
            # function_callしたい関数の情報を取得
            func_name = message["function_call"]["name"]
            arguments = json.loads(message["function_call"]["arguments"])

            # 指定した関数を実行する
            func = getattr(function_calling, func_name)
            function_response = func(arguments.get("item"))

            # 関数の実行結果をAIに返す
            self.past_messages.append({
                "role": "function",
                "name": func_name,
                "content": function_response
                })

            second_answer = openai.ChatCompletion.create(
                model=self.model,
                messages=self.past_messages,
            )

            # AIの返答とトークン数を返す
            answer = second_answer.choices[0].message["content"]
            total_tokens = second_answer.usage["total_tokens"]

            return answer, total_tokens
        else:
            answer = response.choices[0].message["content"]
            total_tokens = response.usage["total_tokens"]

            self.past_messages.append({"role": "assistant", "content": answer})

            return answer, total_tokens

