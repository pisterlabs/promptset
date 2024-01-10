from typing import Any, Protocol, Union
import openai

from enum import Enum

from dataclasses import dataclass



class InvalidTemperature(Exception):
    def __init__(self, temperature : float) -> None:
        self.temperature = temperature
        
        super().__init__(f"temperature must be between 0 and 2, got {temperature}.")


@dataclass
class ChatCompletionResponseChoicesMessage:
    role : str 
    content : str
    
    
@dataclass 
class ChatCompletionResponseChoices:
    index : int
    message : ChatCompletionResponseChoicesMessage
    finish_reason : str
    
    
@dataclass 
class ChatCompletionResponseUsage:
    prompt_tokens : int
    completion_tokens : int
    total_tokens : int


@dataclass
class ChatCompletionResponse:
    id : str 
    object : str
    model : str
    choices : ChatCompletionResponseChoices
    usage : ChatCompletionResponseUsage


GptResponse = dict[str, Any]


class ChatCompletionResponseCreator:
    def create_response_message(self, response : GptResponse) -> ChatCompletionResponseChoicesMessage:
        message = ChatCompletionResponseChoicesMessage(
            role = response['choices'][0]['message']['role'],
            content = response['choices'][0]['message']['content']
        )
        
        return message
    
    def create_response_choice(self, response : GptResponse) -> ChatCompletionResponseChoices:
        choices = ChatCompletionResponseChoices(
            index = response['choices'][0]['index'],
            message = self.create_response_message(response),
            finish_reason=response['choices'][0]['finish_reason']
        )

        return choices
            
        
    def create_response_usage(self, response : GptResponse) -> ChatCompletionResponseUsage:
        usage = ChatCompletionResponseUsage(
            prompt_tokens = response["usage"]["prompt_tokens"],
            completion_tokens = response["usage"]["completion_tokens"],
            total_tokens = response["usage"]["total_tokens"]   
        )

        return usage
        
    
    @classmethod
    def create_response(cls, response : GptResponse) -> ChatCompletionResponse:
        creator = cls()
        
        final_response = ChatCompletionResponse(
            id = response['id'],
            object = response['object'],
            model = response['model'],
            choices = creator.create_response_choice(response),
            usage = creator.create_response_usage(response)
        )
        
        return final_response


class Models(Enum):
    GPT_4 = "gpt-4"
    GPT_4_0613 = "gpt-4-0613"
    GPT_4_32K = "gpt-4-32k"
    GPT_4_32K_0613 = "gpt-4-32k-0613"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_3_5_TURBO_0613 = "gpt-3.5-turbo-0613"
    GPT_3_5_TURBO_16K = "gpt-3.5-turbo-16k"
    GPT_3_5_TURBO_16K_0613 = "gpt-3.5-turbo-16k-0613"


class Roles(Enum):
    SYSTEM = "system"
    USER = "user"



MessageData = dict[str, str]
    
class ChatCompletionMessage(Protocol):
    message : dict[str, str]
    
    def __init__(self, *, content : str) -> None:
        ...
        
    
    def __build(self) -> MessageData:
        ...
    
    
class UserCompletionMessage(ChatCompletionMessage):
    def __init__(self, *, content : str) -> None:
        self.message = self.__build(content=content)
    
    def __build(self, *, content : str) -> MessageData:
        return {"role": Roles.USER.value, "content": content}
    

class SystemCompletionMessage(ChatCompletionMessage):
    def __init__(self, *, content : str) -> None:
        self.message = self.__build(content=content)
    
    
    def __build(self, *, content : str) -> MessageData:
        return {"role": Roles.SYSTEM.value, "content": content}
    


class ChatCompletion:
    def __init__(self, model : Models) -> None:
        self.model = model.value
        self.messages : list[MessageData] = []
        
        self.temperature = 1
        self.n = 1
        self.max_tokens = None
        
        
    def __add_message(self, content : dict[str, str]) -> None:
        self.messages.append(content)

    
    
    def __set_temperature(self, temperature : float) -> None:
        MIN_TEMPERATURE = 0
        MAX_TEMPERATURE = 2
        
        if not MIN_TEMPERATURE <= temperature <= MAX_TEMPERATURE:
            raise InvalidTemperature(temperature)
        
        self.temperature = temperature
        
        
    
    def __generate_params(self) -> dict[str, Union[str, int, float, list[MessageData]]]:
        params = {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
            "n": self.n,
        }
        
        return params
    
    def __get_model_response(self, params : dict[str, Union[str, int, float, list[MessageData]]]) -> GptResponse:
        response = openai.ChatCompletion.create(**params)
        
        return response
    
    def generate(self) -> ChatCompletionResponse:
        params = self.__generate_params()
        response = self.__get_model_response(params)
        
        final_response = ChatCompletionResponseCreator.create_response(response)
        
        
        return final_response
    
    
    
    @classmethod
    def create(
        cls,
        model : Models,
        messages : list[ChatCompletionMessage],
        temperature : float = 1.0,
        n : int = 1,        
    ) -> ChatCompletionResponse:
        chat_completion = ChatCompletion(model = model)
        
        chat_completion.__set_temperature(temperature)
        chat_completion.n = n
        
        for message in messages:
            chat_completion.__add_message(message.message)
            
        return chat_completion.generate()
    
        
    




if __name__ == "__main__":
    pass