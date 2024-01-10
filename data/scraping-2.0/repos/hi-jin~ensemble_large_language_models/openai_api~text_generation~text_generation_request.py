from enum import Enum
import openai
from openai_api.models import Model


class TextGenerationRequestType(Enum):
    CHAT_COMPLETION = "chat_completion"
    COMPLETION = "completion"

class TextGenerationRequest:
    """
    OpenAI API를 통해 텍스트 생성을 요청하는 클래스입니다.\\
    ChatCompletion이나 Completion모두 일관된 인터페이스를 제공하기 위해 이 클래스를 상속받습니다.
    
    usage:
        ```
        api = TextGenerationRequest.get_api(type=TextGenerationRequestType.CHAT_COMPLETION)
        api.request(...)
        ```
    """
    model: Model
    
    def __init__(self):
        print('usage: api = TextGenerationRequest.get_api(type=TextGenerationRequestType.CHAT_COMPLETION)')
        
        raise NotImplementedError("This class is not meant to be instantiated.")
    
    @classmethod
    def get_api(cls, type: TextGenerationRequestType, model: Model):
        cls.type = type
        
        if type == TextGenerationRequestType.CHAT_COMPLETION:
            return _TextGenerationRequestViaChatCompletion(model)
        elif type == TextGenerationRequestType.COMPLETION:
            return _TextGenerationRequestViaCompletion(model)

        raise Exception(f"Invalid type: {type}")
    
    def request(
        self,
        prompt: str,
    ) -> str:
        raise NotImplementedError("This method is not meant to be called.")
    

class _TextGenerationRequestViaChatCompletion(TextGenerationRequest):
    """
    OpenAI API를 통해 텍스트 생성을 요청하는 클래스입니다.\\
    """
    
    def __init__(self, model: Model):
        self.model = model
    
    def request(
        self,
        prompt: str,
    ) -> str:
        message = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=self.model.value,
            messages=message,
        )
        
        result = response["choices"][0]["message"]["content"] # type: ignore
        return result


class _TextGenerationRequestViaCompletion(TextGenerationRequest):
    """
    OpenAI API를 통해 텍스트 생성을 요청하는 클래스입니다.\\
    """
    def __init__(self, model: Model):
        self.model = model
    
    def request(
        self,
        prompt: str,
    ) -> str:
        response = openai.Completion.create(
            model=self.model.value,
            prompt=prompt,
        )
        
        result = response["choices"][0]["text"] # type: ignore
        return result.strip()
    
    
if __name__ == "__main__":
    from core.config import OpenAIConfig
    openai.api_key = OpenAIConfig.API_KEY
    prompt = "1 + 1 = "
    
    api = TextGenerationRequest.get_api(
        type=TextGenerationRequestType.CHAT_COMPLETION,
        model=Model.GPT3_5,
    )
    response = api.request(prompt=prompt)
    print(response)
    
    api = TextGenerationRequest.get_api(
        type=TextGenerationRequestType.COMPLETION,
        model=Model.DAVINCI,
    )
    response = api.request(prompt=prompt)
    print(response)
