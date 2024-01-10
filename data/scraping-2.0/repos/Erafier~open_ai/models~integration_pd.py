from pydantic import BaseModel

from tools import session_project
from pylon.core.tools import log
from ...integrations.models.pd.integration import SecretField
import openai


class IntegrationModel(BaseModel):
    api_token: SecretField | str
    model_name: str = 'text-davinci-003'
    temperature: float = 1.0
    max_tokens: int = 7
    top_p: float = 0.8

    def check_connection(self):
        openai.api_key = self.api_token.unsecret(session_project.get())
        try:
            openai.Model.list()
        except Exception as e:
            log.error(e)
            return str(e)
        return True
