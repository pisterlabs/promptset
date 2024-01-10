import os
from openai import OpenAI
from models.model_wrapper import CausalLanguageModelWrapper
from langchain.chains import ConversationalRetrievalChain


class CausalLanguageModelChatGPT(CausalLanguageModelWrapper):
    # model = "gpt-3.5-turbo"
    model_choices = ['gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-1106', 'gpt-4-1106-preview']

    def __init__(
            self,
            chatgpt_model="gpt-3.5-turbo-1106",
            *args,
            **kwargs
    ):
        super(CausalLanguageModelChatGPT, self).__init__(*args, **kwargs)
        if chatgpt_model not in self.model_choices:
            raise RuntimeError(f'{chatgpt_model} has to be one of the choices {self.model_choices}')
        self.chatgpt_model = chatgpt_model
        self.openai_client = OpenAI(
            api_key=os.environ.get('OPEN_AI_KEY')
        )

    def fine_tune(self):
        raise NotImplemented('This capability has not been implemented yet')

    def call(self, prompt) -> str:
        try:
            completion = self.openai_client.chat.completions.create(
                model=self.chatgpt_model,
                messages=[
                    {'role': 'system', 'content': 'You are a medical professional.'},
                    {'role': 'user', 'content': prompt}
                ],
                max_tokens=self._max_new_tokens
            )
            return completion.choices[0].message.content
        except Exception as e:
            self.get_logger().error(e)
            raise e
