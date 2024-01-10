from typing import Dict, Any, List

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.conversational_retrieval.base import (
    BaseConversationalRetrievalChain,
)
from langchain.schema import Document

from eurelis_kb_framework.acronyms import AcronymsTextTransformer


class AcronymsChainWrapper(BaseConversationalRetrievalChain):
    """
    Wrapper class around a chain to apply an acronyms substitution on a question
    """

    acronyms: AcronymsTextTransformer
    proxy: BaseConversationalRetrievalChain

    def __init__(self, chain: BaseConversationalRetrievalChain, acronyms):
        # not calling the super init is intended
        self.proxy = chain
        self.acronyms = acronyms

    def __getattr__(self, item: str) -> Any:
        """
        Method to forward every access to attribute/method non-present on AcronymsChainWrapper to the proxy instance

        Args:
            item (str): name of an attribute/method

        Returns:
            any
        """
        return getattr(self.proxy, item)

    def _get_docs(
        self,
        question: str,
        inputs: Dict[str, Any],
        *args,
        run_manager: CallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs. Will transform the query using the acronyms text transformer and then call the proxy method"""
        question = self.acronyms.transform(question)
        return self.proxy._get_docs(question, inputs, *args, run_manager=run_manager)

    async def _aget_docs(
        self,
        question: str,
        inputs: Dict[str, Any],
        *,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs. Will transform the query using the acronyms text transformer and then call the proxy method"""
        question = self.acronyms.transform(question)
        return await self.proxy._aget_docs(question, inputs, run_manager=run_manager)
