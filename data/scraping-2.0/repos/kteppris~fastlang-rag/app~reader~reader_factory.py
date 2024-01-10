from omegaconf import OmegaConf
import logging
from langchain.chains.base import Chain
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chains import RetrievalQA

from .chain_kwargs import get_chain_kwargs
from .llm_factory import LLMFactory

logger = logging.getLogger(__name__)

class ReaderFactory:
    """
    Factory for producing reader chains based on configuration.
    """
    @staticmethod
    def get_chain(config: OmegaConf, retriever: VectorStoreRetriever) -> Chain:
        """
        Factory method to get the initialized reader chain based on the provided configuration and retriever.

        Parameters
        ----------
        config : OmegaConf
            Configuration specifying the type of chain and its initialization parameters.
        
        retriever : VectorStoreRetriever
            The retriever instance used for fetching the relevant vectors.

        Returns
        -------
        Chain
            An initialized reader chain. The exact type and behavior depends on the configuration.

        """
        chain_type_kwargs = get_chain_kwargs(config)
        llm = LLMFactory.get_llm(config)
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=config.chain.chain_type,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )
        logger.info("Chain initialized.")
        return chain
