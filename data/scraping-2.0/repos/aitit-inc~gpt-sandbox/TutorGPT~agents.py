from .config import Config
from .messages import Messages
from .counter import token_counter
from .logger import time_logger
import openai


class TutorGPT:
    config = Config()
    messages = Messages()
    conversation_history = []

    # 目次の作成
    @time_logger
    def create_table_of_contents(self, title: str) -> str:
        messages = self.messages.construct_start_messages(title=title)
        # システムプロンプトを除外
        self.human_step(messages[1]["content"])
        tutor_reply = self.create_chat_completion(messages=messages)
        self.tutor_step(tutor_reply=tutor_reply)
        return tutor_reply
    
    # キーワードの作成
    @time_logger
    def create_keywords(self, table_of_contents: str = None) -> str:
        messages = self.messages.keyword_messages(conversation_history=self.conversation_history)
        # システムプロンプトを除外
        self.human_step(messages[1]["content"])
        tutor_reply = self.create_chat_completion(messages=messages)
        self.tutor_step(tutor_reply=tutor_reply)
        return tutor_reply
    
    # 問題文の作成
    @time_logger
    def create_questions(self, chapter: str) -> str:
        messages = self.messages.question_messages(conversation_history=self.conversation_history, chapter=chapter)
        # システムプロンプトを除外
        self.human_step(messages[1]["content"])
        tutor_reply = self.create_chat_completion(messages=messages)
        self.tutor_step(tutor_reply=tutor_reply)
        return tutor_reply


    # 汎用関数
    def create_chat_completion(self, messages: list) -> str:
        num_tokens = token_counter(messages)
        max_tokens = self.config.max_tokens - num_tokens
        response = openai.ChatCompletion.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message["content"]

    def change_model(self, model: str):
        self.config.model = model

    def change_temperature(self, temperature: float):
        self.config.temperature = temperature

    def human_step(self, human_input: str):
        self.conversation_history.append(
            "User: " + human_input + "<END_OF_TURN>"
        )

    def tutor_step(self, tutor_reply: str):
        self.conversation_history.append(
            "Tutor: " + tutor_reply + "<END_OF_TURN>"
        )

    @classmethod
    def from_config(cls,cf):
        return cls(messages=Messages.setup(cf))