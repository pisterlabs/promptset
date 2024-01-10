from typing import Optional, Sequence, Any

from langchain.schema import BaseDocumentTransformer, Document

from eurelis_kb_framework.acronyms import AcronymsTextTransformer


class AcronymsDocumentTransformer(BaseDocumentTransformer):
    """
    Acronyms document transformer, document transformer performing an acronyms transformation
    """

    def __init__(
        self,
        acronyms: AcronymsTextTransformer,
        chain_transformer: Optional[BaseDocumentTransformer] = None,
    ):
        """
        Initializer

        Args:
            acronyms (AcronymsTextTransformer): the acronyms transformer
            chain_transformer: optional transformer to call after acronyms transformation
        """

        self.acronyms = acronyms
        self.chain = chain_transformer

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """
        Transform documents implementation
        """
        # if there is a chain we yield from it, otherwise we directly yield the document (more efficient this way than doing a check in the loop)

        if self.chain:
            for doc in documents:
                new_doc = Document(
                    page_content=self.acronyms.transform(doc.page_content),
                    metadata=doc.metadata.copy(),
                )

                yield from self.chain.transform_documents([new_doc])
        else:
            for doc in documents:
                new_doc = Document(
                    page_content=self.acronyms.transform(doc.page_content),
                    metadata=doc.metadata.copy(),
                )
                yield new_doc

    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        raise NotImplementedError
