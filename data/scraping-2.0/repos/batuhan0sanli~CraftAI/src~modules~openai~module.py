from streamlit import button
from src.clients import OpenAIClient
from src.base.base_module import BaseModule
from src.utils import TextObj, loading
from .schema import ModuleSchema
from .enums import OpenAIEndpoint


class OpenAI(BaseModule):
    schema = ModuleSchema

    def build_method(self) -> None:
        messages = self.data['messages']
        messages = [
            {
                'role': TextObj(message['role'], self.variables)(),
                'content': TextObj(message['content'], self.variables)(),
            } for message in messages]

        response_text = self.data['var'] if 'var' in self.data else None

        if self.data['endpoint'] == OpenAIEndpoint.chat_completion:
            clicked = self.container.button("Send", type="primary")
            if clicked:
                with loading(self.variables, response_text):
                    response = OpenAIClient().chat_completion(messages)
                return response
