from enum import Enum
from typing import Any, Dict, List, Literal, Optional

import yaml
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate, SystemMessage
from langchain.schema.messages import BaseMessage
from pydantic import BaseModel, Field

from shared.llms.openai import GPT_3_5_TURBO


class DefaultAgentTools(Enum):
    GENERALIST_EXPERT = "generalist_expert"
    NO_CODE_PYTHON_FUNCTIONS_EXPERT = "no_code_python_functions_expert"
    THINKER_EXPERT = "thinker_expert"
    SUMMARIZER_EXPERT = "summarize_expert"


class EmbeddingItem(BaseModel):
    content: Optional[str] = None
    folder_path: Optional[str] = None


class Prompts(BaseModel):
    system: str
    tool: Optional[str] = None


EMBEDDINGS_TYPE = Dict[str, Optional[EmbeddingItem]]


class Embeddings(BaseModel):
    __root__: EMBEDDINGS_TYPE


class ExpertItem(BaseModel):
    name: str = "default"
    model: str = GPT_3_5_TURBO
    temperature: float = 0
    max_tokens: Optional[int] = None
    prompts: Optional[Prompts] = Field(
        default=Prompts(system="Hello, I'm a helpful assistant in everything.")
    )
    embeddings: Optional[Embeddings] = None
    use_as_tool: bool = True
    max_tokes_as_tool: int = 200
    model_as_tool: str = GPT_3_5_TURBO
    temperature_as_tool: float = 0.5
    tool_return_direct: bool = False
    query_embeddings_before_ask: bool = True
    create_standalone_question_to_search_context: bool = True
    memory_type: Literal["default", "summary"] = "default"

    def get_chat_messages(self, text) -> List[BaseMessage]:
        template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=self.prompts.system,
                    name=f"{self.name}ExpertGPT",
                    role="system",
                ),
                HumanMessagePromptTemplate.from_template("{text}", role="user"),
            ]
        )

        return template.format_messages(text=text)


EXPERT_TYPE = Dict[str, Optional[ExpertItem]]


class Experts(BaseModel):
    __root__: EXPERT_TYPE


class Planner(BaseModel):
    chain_key: str = "default"
    temperature: float = 0
    max_tokens: Optional[int] = None
    model: str = GPT_3_5_TURBO
    embeddings: Embeddings = Field(
        default=Embeddings(
            __root__=dict(default=EmbeddingItem(content="just a placeholder"))
        )
    )
    get_embeddings_as_tool: bool = True
    save_embeddings_as_tool: bool = True


class Chain(BaseModel):
    chain_key: str = "default"
    temperature: float = 0
    max_tokens: Optional[int] = None
    model: str = GPT_3_5_TURBO
    embeddings: Embeddings = Field(
        default=Embeddings(
            __root__=dict(default=EmbeddingItem(content="just a placeholder"))
        )
    )
    get_embeddings_as_tool: bool = True
    save_embeddings_as_tool: bool = True
    memory_type: Literal["default", "summary"] = "default"


class CustomModule(BaseModel):
    package: str
    attribute: str
    execute: bool = True
    build_attribute: Any = Field(default=None, init=False, private=True)

    def get_attribute_built(self):
        if self.build_attribute is None:
            raise ValueError("Attribute not built")
        return self.build_attribute

    def set_attribute_built(self, value: Any):
        self.build_attribute = value


class CustomTools(CustomModule):
    execute: bool = False


class Config(BaseModel):
    experts: Experts
    chain: Chain = Chain()
    planner: Planner = Planner()
    enabled_default_agent_tools: Optional[List[DefaultAgentTools]] = Field(
        default=None,
    )
    enabled_default_experts: List[DefaultAgentTools] = Field(
        default_factory=lambda: [
            DefaultAgentTools.GENERALIST_EXPERT,
            DefaultAgentTools.NO_CODE_PYTHON_FUNCTIONS_EXPERT,
            DefaultAgentTools.THINKER_EXPERT,
            DefaultAgentTools.SUMMARIZER_EXPERT,
        ]
    )
    custom_tools: Optional[CustomTools] = None


class UIConfig(BaseModel):
    config: Config
    file_name: str
    key: str
    file_path: str


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        return Config(**yaml.safe_load(f.read()))
