from pydantic import BaseModel, Field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from langchain.chains import ConversationalRetrievalChain

class CustomConversationalRetrievalChain(ConversationalRetrievalChain):
    input_docs: Any = Field(None, description="Custom input docs for the retrieval chain")

    def _get_docs(self, *args, **kwargs):
        if self.input_docs is None:
            return super()._get_docs(*args, **kwargs)
        return self.input_docs

    def set_input_docs(self, input_docs):
        if len (input_docs) > 0:
            self.input_docs = input_docs

    