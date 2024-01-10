from langchain.llms import OpenAI, OpenAIChat

from solidchain.configs.config import settings
from solidchain.schemas.text_generation import CausalModel


def get_llm_instance (llm_type: CausalModel):
    match llm_type:
        case CausalModel.TEXT_DAVINCI_003 | CausalModel.TEXT_CURIE_001 | CausalModel.TEXT_BABBAGE_001 | CausalModel.TEXT_ADA_001:
            llm= OpenAI
        case CausalModel.GPT_3_5_TURBO:
            llm= OpenAIChat
        case _:
            raise NotImplementedError
        
    return llm