import openai

from chatbot.summary.summary_base import SummaryBase
from chatbot.global_state import GlobalState

class SummaryOpenai(SummaryBase):
    def init_summarizer(self) -> None:
        gs = GlobalState()
        api_key = gs.config["openai_api_key"]
        openai.api_key = api_key

    def summarize_text(self, text: str) -> str:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that summarizes a conversation between two people."},
                {"role": "user", "content": "Summarize the conversation between two people. Do not continue the "
                                            "conversation!\n" + text}
            ],
            temperature=1
        )

        return completion.choices[0].message["content"]
