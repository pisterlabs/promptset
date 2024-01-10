import langchain
from langchain.cache import SQLiteCache
from langchain.llms import Ollama, OpenAI, VertexAI
from langchain.chat_models import ChatOpenAI, ChatVertexAI

langchain.llm_cache = SQLiteCache(database_path=".langchain.db")


class LanguageModelFactory:
    @staticmethod
    def create_model(model_type, model_name, cache, temperature, api_key=None):
        if model_type.lower() == "ollama":
            return Local(model_type, model_name, temperature, api_key)
        elif model_type.lower() == "openai":
            if api_key is None:
                raise ValueError("External model requires a configuration")
            return OAI(model_type, model_name, cache, temperature, api_key)
        elif model_type.lower() == "vertexai":
            if api_key is None:
                raise ValueError("External model requires a configuration")
            return VAI(model_type, model_name, cache, temperature, api_key)
        elif model_type.lower() == "chatopenai":
            if api_key is None:
                raise ValueError("External model requires a configuration")
            return COAI(model_type, model_name, cache, temperature, api_key)
        elif model_type.lower() == "chatvertexai":
            if api_key is None:
                raise ValueError("External model requires a configuration")
            return CVAI(model_type, model_name, cache, temperature, api_key)
        else:
            raise ValueError(f"Invalid model type: {model_type}")


class BaseModel:
    def __init__(self, model_type, model_name, cache, temperature, apikey):
        self.model_type = model_type
        self.model_name = model_name
        self.cache = cache
        self.temperature = temperature
        self.apikey = apikey

    def run_prompt(self, prompt):
        raise NotImplementedError("Subclasses must implement run_prompt")


class Local(BaseModel):
    def run_prompt(self, prompt):
        return Ollama(model=self.model_name, 
                      temperature=self.temperature)(prompt)


class OAI(BaseModel):
    def run_prompt(self, prompt):
        return OpenAI(
            openai_api_key=self.apikey,
            model=self.model_name,
            temperature=self.temperature,
        )(prompt)


class VAI(BaseModel):
    def run_prompt(self, prompt):
        return VertexAI(
            openai_api_key=self.apikey,
            model=self.model_name,
            temperature=self.temperature,
        )(prompt)


class COAI(BaseModel):
    def run_prompt(self, prompt):
        return ChatOpenAI(
            openai_api_key=self.apikey,
            model=self.model_name,
            temperature=self.temperature,
        )(prompt)
        
class CVAI(BaseModel):
    def run_prompt(self, prompt):
        return ChatVertexAI(
            openai_api_key=self.apikey,
            model=self.model_name,
            temperature=self.temperature,
        )(prompt)

