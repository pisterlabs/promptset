from typing import List, Optional, Dict, Any, Union, Sequence
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManagerForChainRun, BRM, CallbackManager
from langchain.chains.llm import LLMChain
from langchain.llms.openai import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import AgentFinish, AgentAction, BaseMessage, Document, LLMResult
from langchain.schema.output import GenerationChunk, ChatGenerationChunk
from tenacity import RetryCallState

from utils.utils import log_function_call


class MyCallbackManagerForChainRun(CallbackManagerForChainRun):

    def on_chain_end(self, outputs: Union[Dict[str, Any], Any], **kwargs: Any) -> None:
        super().on_chain_end(outputs, **kwargs)

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        super().on_chain_error(error, **kwargs)

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        return super().on_agent_action(action, **kwargs)

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        return super().on_agent_finish(finish, **kwargs)

    def get_child(self, tag: Optional[str] = None) -> CallbackManager:
        return super().get_child(tag)

    def on_text(self, text: str, **kwargs: Any) -> Any:
        return super().on_text(text, **kwargs)

    def on_retry(self, retry_state: RetryCallState, **kwargs: Any) -> None:
        super().on_retry(retry_state, **kwargs)

    def __init__(self, *, run_id: UUID, handlers: List[BaseCallbackHandler],
                 inheritable_handlers: List[BaseCallbackHandler], parent_run_id: Optional[UUID] = None,
                 tags: Optional[List[str]] = None, inheritable_tags: Optional[List[str]] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 inheritable_metadata: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(run_id=run_id, handlers=handlers, inheritable_handlers=inheritable_handlers,
                         parent_run_id=parent_run_id, tags=tags, inheritable_tags=inheritable_tags, metadata=metadata,
                         inheritable_metadata=inheritable_metadata)

    @classmethod
    def get_noop_manager(cls: CallbackManagerForChainRun) -> BRM:
        return super().get_noop_manager()


class MyBaseCallbackHandler(BaseCallbackHandler):

    @property
    def ignore_llm(self) -> bool:
        return super().ignore_llm

    @property
    def ignore_retry(self) -> bool:
        return super().ignore_retry

    @property
    def ignore_chain(self) -> bool:
        return super().ignore_chain

    @property
    def ignore_agent(self) -> bool:
        return super().ignore_agent

    @property
    def ignore_retriever(self) -> bool:
        return super().ignore_retriever

    @property
    def ignore_chat_model(self) -> bool:
        return super().ignore_chat_model

    @log_function_call
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: UUID,
                       parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None,
                       metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        return super().on_chain_start(serialized, inputs, run_id=run_id, parent_run_id=parent_run_id, tags=tags,
                                      metadata=metadata, **kwargs)

    @log_function_call
    def on_text(self, text: str, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        return super().on_text(text, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    @log_function_call
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID,
                     parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None,
                     metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        return super().on_llm_start(serialized, prompts, run_id=run_id, parent_run_id=parent_run_id, tags=tags,
                                    metadata=metadata, **kwargs)

    @log_function_call
    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID] = None,
                   **kwargs: Any) -> Any:
        return super().on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    @log_function_call
    def on_chain_end(self, outputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Optional[UUID] = None,
                     **kwargs: Any) -> Any:
        return super().on_chain_end(outputs, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    @log_function_call
    def on_llm_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID] = None,
                     **kwargs: Any) -> Any:
        return super().on_llm_error(error, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    @log_function_call
    def on_chain_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID] = None,
                       **kwargs: Any) -> Any:
        return super().on_chain_error(error, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    @log_function_call
    def on_llm_new_token(self, token: str, *, chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
                         run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        return super().on_llm_new_token(token, chunk=chunk, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    @log_function_call
    def on_agent_action(self, action: AgentAction, *, run_id: UUID, parent_run_id: Optional[UUID] = None,
                        **kwargs: Any) -> Any:
        return super().on_agent_action(action, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    @log_function_call
    def on_agent_finish(self, finish: AgentFinish, *, run_id: UUID, parent_run_id: Optional[UUID] = None,
                        **kwargs: Any) -> Any:
        return super().on_agent_finish(finish, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    @log_function_call
    def on_tool_end(self, output: str, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        return super().on_tool_end(output, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    @log_function_call
    def on_tool_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID] = None,
                      **kwargs: Any) -> Any:
        return super().on_tool_error(error, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    @log_function_call
    def on_retriever_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID] = None,
                           **kwargs: Any) -> Any:
        return super().on_retriever_error(error, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    @log_function_call
    def on_retriever_end(self, documents: Sequence[Document], *, run_id: UUID, parent_run_id: Optional[UUID] = None,
                         **kwargs: Any) -> Any:
        return super().on_retriever_end(documents, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    @log_function_call
    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], *, run_id: UUID,
                            parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None,
                            metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        pass

    @log_function_call
    def on_retriever_start(self, serialized: Dict[str, Any], query: str, *, run_id: UUID,
                           parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None,
                           metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        return super().on_retriever_start(serialized, query, run_id=run_id, parent_run_id=parent_run_id, tags=tags,
                                          metadata=metadata, **kwargs)

    @log_function_call
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, *, run_id: UUID,
                      parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None,
                      metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        return super().on_tool_start(serialized, input_str, run_id=run_id, parent_run_id=parent_run_id, tags=tags,
                                     metadata=metadata, **kwargs)

    @log_function_call
    def on_retry(self, retry_state: RetryCallState, *, run_id: UUID, parent_run_id: Optional[UUID] = None,
                 **kwargs: Any) -> Any:
        return super().on_retry(retry_state, run_id=run_id, parent_run_id=parent_run_id, **kwargs)


def test():
    llm = OpenAI()
    prompt = PromptTemplate.from_template("如何浇花")
    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    predict = llm_chain(inputs={}, callbacks=[MyBaseCallbackHandler(), ], )
    print(predict)


@log_function_call
def hello():
    pass


if __name__ == '__main__':
    test()
