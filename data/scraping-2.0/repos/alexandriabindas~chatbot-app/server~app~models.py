from enum import Enum

from pydantic import BaseModel, Field
from langchain.embeddings import GPT4AllEmbeddings, OpenAIEmbeddings, OllamaEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader, PyPDFLoader


class User(BaseModel):
    _id: str | int
    name: str


class File(BaseModel):
    size: int
    fileCopyUri: str | None = None
    name: str
    uri: str
    local_uri: str
    type: str


class Message(BaseModel):
    room: str = 'default'
    active_bots: list[str] = Field(default_factory=list)
    user: User
    prompt: str
    system: bool = False


class ToClientEvents(Enum):
    TYPING = 'typing'
    MESSAGE = 'message'
    CONFIG_UPDATE = 'config_update'


# TODO: Move these
embeddings = {
    'gpt-4': {
        'title': 'GPT4AllEmbeddings',
        'value': 'gpt-4',
        'class': GPT4AllEmbeddings
    },
     'ollama': {
        'title': 'OllamaEmbeddings',
        'value': 'ollama',
        'class': OllamaEmbeddings
    },
    'openai': {
        'title': 'OpenAIEmbeddings',
        'value': 'openai',
        'class': OpenAIEmbeddings
    }
}

loaders = {
    'default': {
        'title': 'PyPDFLoader',
        'value': 'default',
        'config': PyPDFLoader
    },
    'unstruct': {
        'title': 'UnstructuredPDFLoader',
        'value': 'unstruct',
        'config': UnstructuredPDFLoader
    }
}