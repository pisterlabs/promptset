import openai
import tiktoken
from nonebot import get_driver
from .characterSettings import character_settings

config = get_driver().config


class ChatHandler:
    # 记录各个人对话的字典
    transcripts_of_conversations = {}
    default_model = "gpt-3.5-turbo"

    def __init__(self):
        global config
        openai.api_key = config.openai_api_key

    def ask(self, message: str, mode: str, user_id: int):
        chat_api_res_message = "似乎出了一些问题，无法回复。"
        Characterisation = character_settings[mode] if mode in character_settings else character_settings["默认"]
        if mode not in self.transcripts_of_conversations:
            self.transcripts_of_conversations[mode] = {}
        if user_id not in self.transcripts_of_conversations[mode]:
            self.transcripts_of_conversations[mode][user_id] = [{"role": "system", "content": Characterisation}]
            self.transcripts_of_conversations[mode][user_id].append({"role": "user", "content": message})
        else:
            self.transcripts_of_conversations[mode][user_id].append({"role": "user", "content": message})
        try:
            chat_api_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=self.transcripts_of_conversations[mode][user_id]
            )
        except Exception as error:
            print(error)
            chat_api_response = None
            del self.transcripts_of_conversations[mode][user_id]
            chat_api_res_message += "\n\n可能由于积累对话过长，现在已经将您与茗懿的对话记录重置。"

        if chat_api_response is not None:
            chat_api_res_message = chat_api_response["choices"][0]["message"]["content"]
            self.transcripts_of_conversations[mode][user_id].append(
                {"role": "assistant", "content": chat_api_res_message}
            )
        # 如果数组长度超过50，则删除该键值对
        if len(self.transcripts_of_conversations[mode][user_id]) > 50:
            del self.transcripts_of_conversations[mode][user_id]
            chat_api_res_message += "\n\n为了节省内存，主人设定超过50轮对话将强制清空对话记录，现在已经将您与茗懿的对话记录重置。"
        print(chat_api_res_message)
        return chat_api_res_message

    def clean_conversations(self, mode: str, user_id: int):
        if mode in self.transcripts_of_conversations:
            if user_id in self.transcripts_of_conversations[mode]:
                del self.transcripts_of_conversations[mode][user_id]
                return True
        return False

    def clean_all_conversations(self):
        self.transcripts_of_conversations = {}
        return True

    def num_tokens_from_messages(self, message, mode, user_id, model="gpt-3.5-turbo"):
        temp_toc = self.transcripts_of_conversations
        Characterisation = character_settings[mode] if mode in character_settings else character_settings["默认"]
        if mode not in temp_toc:
            temp_toc[mode] = {}
        if user_id not in temp_toc[mode]:
            temp_toc[mode][user_id] = [{"role": "system", "content": Characterisation}]
            temp_toc[mode][user_id].append({"role": "user", "content": message})
        else:
            temp_toc[mode][user_id].append({"role": "user", "content": message})
        # 计算对话的token数量
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = 0
        for message in temp_toc[mode][user_id]:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
