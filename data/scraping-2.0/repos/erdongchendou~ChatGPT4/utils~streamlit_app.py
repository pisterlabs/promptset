from pathlib import Path
from datetime import datetime

import streamlit as st

from utils.role_definition import role_definitions
from config.personal_info import introduction
from ai_bots.openai_bot import OpenAIBot


class ModelsOption:
    tts_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    dalle_models = ["dall-e-2", "dall-e-3"]
    dalle2_sizes = ["256x256", "512x512", "1024x1024"]
    dalle3_sizes = ["1024x1024", "1792x1024", "1024x1792"]
    dalle_quality = ["standard", "hd"]


class StreamlitApp:
    def __init__(self):
        self.ai_bot = OpenAIBot()
        self.st = st

        self.set_title()
        self.choose_mode()
        self.init_messages()
        self.show_messages()

    def init_messages(self):
        if "messages" not in self.st.session_state:
            chat_mode = self.st.session_state.mode
            self.st.session_state["messages"] = []
            self.st.session_state["messages"].extend(role_definitions[chat_mode])
        return None

    def set_title(self):
        self.st.title("ChatGPT4")
        self.st.caption(
            (
                "利用Python调用GPT4的接口, 开发的ChatGPT。 \n\n"
                "作者陈雷雷，加我微信：aierdongboxer ， 免费学习《零基础python入门课程》。"
            )
        )
        return None

    def choose_mode(self):
        with self.st.sidebar:
            self.st.session_state.mode = self.st.selectbox(
                "聊天模式",
                list(role_definitions.keys()),
                on_change=lambda: self.st.session_state.clear(),
            )

            if self.st.session_state.mode == "英语口语练习":
                self.options_for_english_practice()

            if self.st.session_state.mode == "画图":
                self.options_for_generate_image()

            if self.st.session_state.mode == "文字转语音":
                self.options_for_english_practice()

            self.st.write(introduction)

        return None

    def options_for_generate_image(self):
        self.ai_bot.draw_model = self.st.selectbox("模型", ModelsOption.dalle_models)
        if self.ai_bot.draw_model == "dall-e-2":
            self.st.session_state.image_size = self.st.selectbox(
                "分辨率", ModelsOption.dalle2_sizes
            )
            self.st.session_state.number_of_choices = self.st.selectbox(
                "生成个数", list(range(1, 10))
            )
            self.st.session_state.image_quality = "standard"
        else:
            self.st.session_state.image_size = self.st.selectbox(
                "分辨率", ModelsOption.dalle3_sizes
            )
            self.st.session_state.number_of_choices = 1
            self.st.session_state.image_quality = self.st.selectbox(
                "画质", ModelsOption.dalle_quality
            )

    def options_for_english_practice(self):
        self.st.session_state.number_of_choices = self.st.selectbox(
            "生成个数", list(range(1, 10))
        )
        self.st.session_state.voice = self.st.selectbox("声音", ModelsOption.tts_voices)
        return None

    def show_messages(self):
        for msg in self.st.session_state.messages:
            if msg["role"] == "system":
                continue

            if "img" in msg["content"] and "https:" in msg["content"]:
                self.st.chat_message(msg["role"]).image(msg["content"])
            elif "mp3" in msg["content"]:
                self.st.chat_message(msg["role"]).audio(msg["content"])
            else:
                self.st.chat_message(msg["role"]).write(msg["content"])
        return None

    def chat(self):
        if prompt := self.st.chat_input():
            self.st.session_state.messages.append({"role": "user", "content": prompt})
            self.st.chat_message("user").write(prompt)

            if self.st.session_state.mode == "聊天":
                self.generate_chat()

            elif self.st.session_state.mode == "画图":
                self.generate_image()

            elif self.st.session_state.mode == "英语口语练习":
                self.english_practice()

            elif self.st.session_state.mode == "文字转语音":
                self.text_to_speech()
        return None

    def generate_chat(self):
        with self.st.chat_message("assistant"):
            message_placeholder = self.st.empty()
            full_response = ""

        for content in self.ai_bot.stream_chat(st):
            full_response += content
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)
        self.st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
        return None

    def generate_image(self):
        image_infos = self.ai_bot.generate_image(st)
        for data in image_infos.data:
            self.st.chat_message("assistant").image(data.url)

            self.st.session_state.messages.append(
                {"role": "assistant", "content": data.url}
            )
        return None

    def english_practice(self):
        translations = self.chinese_to_english()
        self.generate_audio(translations)
        return None

    def chinese_to_english(self):
        rspd_translation = self.ai_bot.chat(self.st)
        translations = []
        for rspd in rspd_translation.choices:
            translations.append(rspd.message.content)
        return translations

    def generate_audio(self, contents, show_text=True):
        date = datetime.now().strftime("%Y%m%d")
        Path(date).mkdir(parents=True, exist_ok=True)

        for index, content in enumerate(contents):
            datetime_with_second = datetime.now().strftime("%Y%m%d_%H%M%S")
            speech_file_path = f"{date}/{datetime_with_second}.mp3"

            if show_text:
                self.st.session_state.messages.append(
                    {"role": "assistant", "content": content}
                )
                rspd_with_num = f"{index + 1}.{content}\n\n"
                self.st.chat_message("assistant").empty().markdown(rspd_with_num)

            rspd_audio = self.ai_bot.generate_audio(st)
            rspd_audio.stream_to_file(speech_file_path)

            self.st.chat_message("assistant").audio(speech_file_path)

            self.st.session_state.messages.append(
                {"role": "assistant", "content": speech_file_path}
            )
        return None

    def text_to_speech(self):
        contents = [self.st.session_state.messages[-1]["content"]]
        self.generate_audio(contents, show_text=False)
        return None
