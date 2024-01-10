from abc import abstractmethod, ABC

from llama_index import ServiceContext, LLMPredictor, LangchainEmbedding

from core.lifecycle import Lifecycle
from langchain_manager.manager import BaseLangChainManager


# def get_callback_manager() -> CallbackManager:
#     from llama_index.callbacks import (
#         WandbCallbackHandler,
#         CallbackManager,
#         LlamaDebugHandler,
#     )
#     llama_debug = LlamaDebugHandler(print_trace_on_end=True)
#     # wandb.init args
#     run_args = dict(
#         project="llamaindex",
#     )
#     wandb_callback = WandbCallbackHandler(run_args=run_args)
#     return CallbackManager([llama_debug, wandb_callback])


class ServiceContextManager(Lifecycle, ABC):
    @abstractmethod
    def get_service_context(self) -> ServiceContext:
        pass


class AzureServiceContextManager(ServiceContextManager):
    lc_manager: BaseLangChainManager
    service_context: ServiceContext

    def __init__(self, lc_manager: BaseLangChainManager):
        super().__init__()
        self.lc_manager = lc_manager

    def get_service_context(self) -> ServiceContext:
        if self.service_context is None:
            raise ValueError(
                "service context is not ready, check for lifecycle statement"
            )
        return self.service_context

    def do_init(self) -> None:
        # define embedding
        embedding = LangchainEmbedding(self.lc_manager.get_embedding())
        # define LLM
        llm_predictor = LLMPredictor(llm=self.lc_manager.get_llm())
        # configure service context
        self.service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor,
            embed_model=embedding,
            # callback_manager=get_callback_manager(),
        )

    def do_start(self) -> None:
        self.logger.info(
            "[do_start][embedding] last used usage: %d",
            self.service_context.embed_model.total_tokens_used,
        )
        self.logger.info(
            "[do_start][predict] last used usage: %d",
            self.service_context.llm_predictor.total_tokens_used,
        )

    def do_stop(self) -> None:
        self.logger.info(
            "[do_stop][embedding] last used usage: %d",
            self.service_context.embed_model.total_tokens_used,
        )
        self.logger.info(
            "[do_stop][predict] last used usage: %d",
            self.service_context.llm_predictor.total_tokens_used,
        )

    def do_dispose(self) -> None:
        self.logger.info(
            "[do_dispose] total used token: %d",
            self.service_context.llm_predictor.total_tokens_used,
        )


class HuggingFaceChineseOptServiceContextManager(ServiceContextManager):
    lc_manager: BaseLangChainManager
    service_context: ServiceContext

    def __init__(self, lc_manager: BaseLangChainManager):
        super().__init__()
        self.lc_manager = lc_manager

    def get_service_context(self) -> ServiceContext:
        if self.service_context is None:
            raise ValueError(
                "service context is not ready, check for lifecycle statement"
            )
        return self.service_context

    def do_init(self) -> None:
        # define embedding
        from langchain.embeddings import HuggingFaceEmbeddings

        model_name = "GanymedeNil/text2vec-large-chinese"
        hf_embedding = HuggingFaceEmbeddings(
            model_name=model_name, model_kwargs={"device": "cpu"}
        )

        embedding = LangchainEmbedding(hf_embedding)
        # define LLM
        llm_predictor = LLMPredictor(self.lc_manager.get_llm())
        # configure service context
        self.service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor,
            embed_model=embedding,
            # callback_manager=get_callback_manager()
        )

    def do_start(self) -> None:
        self.logger.info(
            "[do_start][embedding] last used usage: %d",
            self.service_context.embed_model.total_tokens_used,
        )
        self.logger.info(
            "[do_start][predict] last used usage: %d",
            self.service_context.llm_predictor.total_tokens_used,
        )

    def do_stop(self) -> None:
        self.logger.info(
            "[do_stop][embedding] last used usage: %d",
            self.service_context.embed_model.total_tokens_used,
        )
        self.logger.info(
            "[do_stop][predict] last used usage: %d",
            self.service_context.llm_predictor.total_tokens_used,
        )

    def do_dispose(self) -> None:
        self.logger.info(
            "[do_dispose] total used token: %d",
            self.service_context.llm_predictor.total_tokens_used,
        )
