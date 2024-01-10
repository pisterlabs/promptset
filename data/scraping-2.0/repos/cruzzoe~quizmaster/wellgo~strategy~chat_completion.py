import openai
import logging
from . import BaseStrategy

logger = logging.getLogger(__name__)


class ChatCompletionStrategy(BaseStrategy):
    def __init__(self, model='gpt-4'):
        self.name = "chat_completion"
        self.model = model
        self.client = openai.OpenAI()

    def determine_answer(self, qu, choices):
        """Use ChatGPT to solve qu and return answer from answers"""

        response = self.client.chat.completions.create(
            model=self.model, 
              messages=[
                {"role": "system", "content": "Give Response as a single letter from options that answers this question."},
                {"role": "user", "content": f"""
                Q: {qu}
                Choices: {choices}
                 """},
            ]
        )
        answer = response.choices[0].message.content
        answer = answer.strip("\n").strip(" ")[0]
        if answer not in ["A", "B", "C", "D"]:
            raise Exception(f"Wrong Choice! Investigate ChatGPT response... {answer}")
        return answer
