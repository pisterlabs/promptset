import json
from datetime import datetime

import openai
import tiktoken
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import (ChatMessageHistory,
                              ConversationSummaryBufferMemory)
from langchain.schema import messages_from_dict, messages_to_dict


class GptChat:
    def __init__(self, ai_chara, ai_dialogues):
        self.is_first_call = True
        self.chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)

        system_prompt = f"""
        あなたは次の回答から以下の設定に基づいたキャラクターになりきってロールプレイし、私と会話して下さい。
        {ai_chara}
        回答は20文字前後の短いセリフで返してください。
        """

        ai_prompt = f"""
        以下にセリフ例を示します。
        {ai_dialogues}
        """

        self.history = ChatMessageHistory()
        self.history.add_user_message(system_prompt)
        self.history.add_ai_message(ai_prompt)

        self.memory = ConversationSummaryBufferMemory(
            llm=self.chat,
            max_token_limit=2000,
            chat_memory=self.history,
            return_messages=True)

        self.conversation = ConversationChain(
            llm=self.chat,
            memory=self.memory,
            verbose=True,
        )

    def gptResponse(self, user_input):
        print("GPT3モデル識別中")
        if self.is_first_call ==  True:
            self.is_first_call = False

        # ユーザーの入力をmessagesに追加
        return_msg = self.conversation.predict(input=user_input)

        # print("使用トークン数:", response['usage']['total_tokens'])
        return return_msg

    def saveConversation(self, ai_name):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_dicts = messages_to_dict(self.history.messages)
        filename = f"./log/{ai_name}_{current_time}.json"

        with open(filename, "w", encoding='utf-8') as f:
            json.dump(history_dicts, f, ensure_ascii=False)

    def summarizeConversation(self, summary, ai_name):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./log/{ai_name}_{current_time}_summary.log"
        with open(filename, "w", encoding='utf-8') as f:
            f.write(summary)

    def loadPreviousChat(self, log_file_path):
        with open(log_file_path, "r", encoding="utf-8") as f:
            previous_messages = json.load(f)
        self.messages.extend(previous_messages)
    
    def char_completion(self, messages):
        print("GPT3モデル識別中")

        # ユーザーの入力をmessagesに追加
        self.messages += messages

        # GPT-3による応答を生成
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=self.messages,
            temperature=0.5,
        )

        return_msg = response['choices'][0]["message"]["content"]
        print("使用トークン数:", response['usage']['total_tokens'])
        
        return return_msg

    def gptConcat(self, voice_msg, return_msg):
        # ユーザーの入力をmessagesに追加
        self.messages.append({"role": "user", "content": voice_msg})
        # GPT-3の応答をmessagesに追加
        self.messages.append({"role": "assistant", "content": return_msg})

# ChatGPT API
def completion(system, user, model):
    # プロンプト設定
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

    print("GPT3モデル識別中")

    # GPT-3による応答を生成
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.5,
    )

    # GPT-3の応答を取得
    return_msg = response['choices'][0]["message"]["content"]
    print(return_msg)
    print("使用トークン数:", response['usage']['total_tokens'])

    return return_msg
