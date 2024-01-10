import openai

class Config:
    openai_key = "sk-"
    serpapi_key = ""
    
    @staticmethod
    def get_openai_key():
        return Config.openai_key
    
    @staticmethod
    def get_serpapi_key():
        return Config.serpapi_key


