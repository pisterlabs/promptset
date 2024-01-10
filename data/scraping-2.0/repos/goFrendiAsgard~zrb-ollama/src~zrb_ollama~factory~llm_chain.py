from langchain.chains import LLMChain
from .schema import LLMChainFactory
from ..task.any_prompt_task import AnyPromptTask


def llm_chain_factory(verbose: bool = False) -> LLMChainFactory:
    def create_llm_chain(task: AnyPromptTask) -> LLMChain:
        return LLMChain(
            llm=task.get_chat_model(),
            prompt=task.get_chat_prompt_template(),
            memory=task.get_chat_memory(),
            verbose=verbose
        )
    return create_llm_chain
