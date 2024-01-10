import openai


from typing import Dict, List, Union, Callable
from dataclasses import dataclass


PROVIDER_LIST = ["openai", "kakao"]
MessageType = List[Dict[str, str]]


@dataclass
class ChatModelRespondType:
    """
    respond : GPT에 넣었을떄 나오는 응답(string)
    respond_with_message : system, user, assistant를 모두 포함(MessageType)
    info : Token 사용량과 관련된 정보(dict)
    """

    respond: str
    respond_with_message: MessageType
    info: Dict


class ChatModel:
    # 밑에 모델을 편하게 선택할 수 있는 get_model_name 클래스가 있지만,
    # 여기서도 바로 값을 넣어 api를 구동할 수 있음.
    # 알맞은 모델이름을 넣고 클래스 선언.
    # 함수들에 딕셔너리 형태의 메세지를 넣으면 api값을 받아옴.
    # API 호출시 필수 파라미터가 아닌 값들은 일단 주석처리해둠.
    def __init__(
        self,
        model_type: str,
        temperature: float = 1.0,
        stable: bool = False,
        more_tokens: bool = False,
        provider: str = "openai",
    ):
        assert provider in PROVIDER_LIST, f"Provider Not Supported.(supported : {str(PROVIDER_LIST)})"
        self.model_type = model_type
        self.temperature = temperature

        if provider == "openai":
            from provider.openai import OpenaiProvider, getModelName

            self.getModelName = getModelName
            self.provider = OpenaiProvider()

        self.model = self.getModelName(model_type, stable, more_tokens)

    def __get_provider(self, messages: MessageType):
        return self.provider.get(messages, self.model, self.temperature)

    def create_default_message(self, user_input: str) -> MessageType:
        return self.create_message(user_input)

    def create_message(self, user_input: str) -> MessageType:
        return [
            {
                "role": "user",
                "content": user_input,
            },
        ]

    def get_reply(self, messages: MessageType) -> ChatModelRespondType:
        respond = self.__get_provider(messages)
        return ChatModelRespondType(
            respond=respond.respond,
            respond_with_message=[*messages, [respond.message]],
            info={**respond.info},
        )

    def forward(
        self, user_input: str, get_system_prompt: Union[Callable[[], str], None] = None
    ) -> ChatModelRespondType:
        if get_system_prompt is None:
            message = self.create_default_message(user_input)
        else:
            message = self.create_message(get_system_prompt(), user_input)
        reply = self.get_reply(message)
        return reply


class OpenAIEmbeddingModel:
    # 워드임베딩 api는 하나밖에 없으므로 모델명은 인풋으로 받지 않음.
    # 인풋문장을 넣어주면 출력되는 get_embedding 함수가 있음.

    def get_embeddings(self, input):
        embeddings = openai.Embedding.create(
            model="text-embedding-ada-002", input=input
        )
        return embeddings

    def get_vector(self, input):
        return self.get_embeddings(input).data[0].embedding


def getAvailable():
    "Return all available models."
    return openai.Model.list()
