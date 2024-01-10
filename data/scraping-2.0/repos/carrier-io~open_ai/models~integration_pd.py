import json
from typing import List, Optional
from pydantic import BaseModel, root_validator, validator

from tools import session_project, rpc_tools, VaultClient
from pylon.core.tools import log
from ...integrations.models.pd.integration import SecretField


def get_capabilities_map():
    vault_client = VaultClient()
    secrets = vault_client.get_all_secrets()
    return json.loads(secrets.get('open_ai_capatibilities_map', ''))


def get_token_limits():
    vault_client = VaultClient()
    secrets = vault_client.get_all_secrets()
    return json.loads(secrets.get('open_ai_token_limits', ''))


class CapabilitiesModel(BaseModel):
    completion: bool = False
    chat_completion: bool = False
    embeddings: bool = False

class AIModel(BaseModel):
    id: str
    name: str
    capabilities: dict = {}
    token_limit: Optional[int]

    @validator('name', always=True, check_fields=False)
    def name_validator(cls, value, values):
        return values.get('id', value)

    @validator('capabilities', always=True, check_fields=False)
    def capabilities_validator(cls, value, values):
        if value:
            return value
        capabilities = CapabilitiesModel()
        if capabilities_map := get_capabilities_map():
            for capability, models in capabilities_map.items():
                if any([model == values['id'] for model in models]):
                    setattr(capabilities, capability, True)
        return capabilities

    @validator('token_limit', always=True, check_fields=False)
    def token_limit_validator(cls, value, values):
        if value:
            return value
        token_limits = get_token_limits()
        return token_limits.get(values.get('id'), 8096)

class IntegrationModel(BaseModel):
    api_token: SecretField | str
    model_name: str = 'text-davinci-003'
    models: List[AIModel] = []
    api_version: str | None = None
    api_base: str = "https://api.openai.com/v1"
    api_type: str = "open_ai"
    temperature: float = 1.0
    max_tokens: int = 7
    top_p: float = 0.8

    @root_validator(pre=True)
    def prepare_model_list(cls, values):
        models = values.get('models')
        if models and isinstance(models[0], str):
            values['models'] = [AIModel(id=model, name=model).dict(by_alias=True) for model in models]
        return values

    @property
    def token_limit(self):
        return next((model.token_limit for model in self.models if model.id == self.model_name), 8096)

    def get_token_limit(self, model_name):
        return next((model.token_limit for model in self.models if model.id == model_name), 8096)

    def check_connection(self):
        from openai import Model
        api_key = self.api_token.unsecret(session_project.get())
        api_type = self.api_type
        api_version = self.api_version
        api_base = self.api_base
        try:
            Model.list(
                api_key=api_key, api_base=api_base, api_type=api_type, api_version=api_version
                )
        except Exception as e:
            log.error(e)
            return str(e)
        return True

    def refresh_models(self, project_id):
        integration_name = 'open_ai'
        payload = {
            'name': integration_name,
            'settings': self.dict(),
            'project_id': project_id
        }
        return getattr(rpc_tools.RpcMixin().rpc.call, f'{integration_name}_set_models')(payload)


class OpenAISettings(BaseModel):
    model_name: str = 'text-davinci-003'
    temperature: float = 1.0
    max_tokens: int = 7
    top_p: float = 0.8
