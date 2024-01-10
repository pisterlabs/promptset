from langchain.chat_models.gigachat import GigaChat
from langchain.schema import HumanMessage, SystemMessage

from config.config import SBER_API_KEY

MODEL = "gpt-3.5-turbo-1106"
MAX_WORDS = 1000  # max_tokens = 2048


sys_prompt = """Твоя задача выделить главную информацию из записи совещания.
            Выдели несколько главных тезисов, которые обсуждались и перечисли их.
            Постарайся сохранить важные детали.
            Твой ответ должен содержать ТОЛЬКО ТЕЗИСЫ, в формате "{порядковый_номер}.{Тезис}\n".
            """

optional_prompt = """Твоя задача составить ОДИН вопрос с ЧЕТЫРЬМЯ вариантами ответа по записи семинара. 
                Ты должен предложить правильный вариант ответа. 
                Результат должен соответствовать следующим требованиям:
                1) Первый абзац - текст вопроса.
                2) Четыре следующих абзаца - варианты ответов.
                3) Последний абзац - правильный ответ."""


class Summarizer:
    def __init__(self, combined_text):
        self.texts = [text for (text, speaker) in combined_text]
        self.speakers = [speaker for (text, speaker) in combined_text]
        self.chat = GigaChat(credentials=SBER_API_KEY, verify_ssl_certs=False)

    def summarize(self):
        summaries = self.get_summary()
        size = len(summaries)
        summary = ""
        for i, part in enumerate(summaries):
            summary += f"Часть {i + 1} из {size}:\n"
            if part.split("\n")[0] == "Тезисы:":
                part = "\n".join(part.split("\n")[1:])
            summary += part + "\n\n"
        return summary

    def get_summary(self):
        texts = self.prepare_texts()
        result = []
        for text in texts:
            prompt = text
            try:
                response = self.get_completion(prompt)
            except Exception:
                response = self.get_completion(prompt)
            result.append(response)
        return result

    def prepare_texts(self):
        curr_size = 0
        curr_text = ""
        text = []
        for line in self.texts:
            num_words = len(line.split(" "))
            if num_words + curr_size < MAX_WORDS:
                curr_text += line
                curr_size += num_words
            else:
                text.append(curr_text)
                curr_text = line
                curr_size = num_words
        text.append(curr_text)
        return text

    def get_completion(self, prompt, system_prompt=sys_prompt, model=MODEL):
        messages = [SystemMessage(content=system_prompt)]
        messages.append(HumanMessage(content=prompt))
        response = self.chat(messages)
        return response.content

    def get_question(self):
        texts = self.prepare_texts()
        result = []
        for text in texts:
            prompt = text
            try:
                response = self.get_completion(prompt=prompt, system_prompt=optional_prompt)
            except Exception:
                response = self.get_completion(prompt=prompt, system_prompt=optional_prompt)
            result.append(response)
        return result

    def get_questions(self):
        questions = self.get_question()
        size = len(questions)
        quest = ""
        for i, part in enumerate(questions):
            #summary += f"Часть {i + 1} из {size}:\n"
            quest += part + "\n\n"
        return quest
