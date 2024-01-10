import os

import openai
from dotenv import load_dotenv


class ChatAPI:
    def __init__(self, api_key, context):
        openai.api_key = api_key
        self.context = context

    def post(self, utterance: str):
        self.context.append({'role': 'assistant', 'content': utterance})
        self.context.append({'role': 'user', 'content': utterance})
        r = self._completion()
        reply = r.choices[0].message['content']
        self.context.append({'role': 'assistant', 'content': reply})
        return reply

    def _completion(self):
        # https://platform.openai.com/docs/api-reference/chat/create
        r = openai.ChatCompletion.create(
            # NOTE: gpt-3.5-turboは小規模で速度が速く、比較的簡単な自然言語処理タスクに使用されることが多い
            # see: https://platform.openai.com/docs/models/gpt-4
            # pricing: https://openai.com/pricing
            model="gpt-3.5-turbo-0613",  # https://platform.openai.com/docs/models
            messages=self.context,
        )
        return r


if __name__ == '__main__':
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

    system_content = '''
        # Instructions :
        You are an American professional English teacher.
        Please chat with me under the following constraints.

        # Constraints:

        I am a beginner in English.
        You can choose the topic for our conversation.
        We will take turns writing one sentence at a time.
        If you notice any grammatical errors in my sentences,
        please correct them and explain why you made the correction.
        Please respond in 30 words or less.
        '''

    system_context = {"role": "system", "content": system_content}

    # テスト用の会話コンテキスト
    conversation_context = [system_context]

    conversation = ChatAPI(api_key, context=conversation_context)
    r = conversation.post("I'm fine. I work a lot.")
    print(r)
