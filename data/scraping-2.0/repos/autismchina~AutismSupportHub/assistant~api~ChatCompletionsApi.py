from . OpenAIApi import OpenAIApi
from .OpenAIConfig import OpenAIConfig

class ChatCompletionsApi(OpenAIApi):
    def __init__(self, api_key: str, organization: str = None) -> None:
        super().__init__(api_key, organization)

    def create_chat_completions(self,config:OpenAIConfig,message_placeholder):
        full_response = ""
        for response in self.client.chat.completions.create(
            model=config.model,
            messages=config.messages,
            stream=True
        ):
            full_response += (response.choices[0].delta.content or "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        return full_response
   
