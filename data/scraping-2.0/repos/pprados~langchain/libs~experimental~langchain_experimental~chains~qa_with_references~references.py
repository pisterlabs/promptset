import logging
import re
from typing import Set

from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Extra
from langchain.schema import BaseOutputParser

logger = logging.getLogger(__name__)

# To optimize the consumption of tokens, it's better to use only 'text', without json.
# Else the schema consume ~300 tokens and the response 20 tokens by step
_OPTIMIZE = True  # Experimental


class References(BaseModel):
    """
    Response and referenced documents.
    """

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    response: str
    """ The response """
    documents_ids: Set[str] = set()
    """ The list of documents used to response """

    def __str__(self) -> str:
        if _OPTIMIZE:
            return f'{self.response}\nIDX:{",".join(map(str, self.documents_ids))}'
        else:
            return self.json()


references_parser: BaseOutputParser
if _OPTIMIZE:

    class _ReferencesParser(BaseOutputParser):
        """An optimised parser for Reference.
        It's more effective than the pydantic approach
        """

        @property
        def lc_serializable(self) -> bool:
            return True

        @property
        def _type(self) -> str:
            """Return the type key."""
            return "reference_parser"

        def get_format_instructions(self) -> str:
            return (
                "Your response should be in the form:\n"
                "Answer:the response\n"
                "IDX: a comma-separated list of document identifiers used "
                "in the response"
            )

        def parse(self, text: str) -> References:
            regex = r"(?:Answer:)?(.*)\nIDX:(.*)"
            match = re.search(regex, text)
            if match:
                ids: Set[int] = set()
                for str_doc_id in match[2].split(","):
                    m = re.match(r"(?:_idx_)?(\d+)", str_doc_id.strip())
                    if m:
                        ids.add(int(m[1]))

                return References(response=match[1].strip(), documents_ids=ids)
            else:
                raise ValueError(f"Could not parse output: {text}")

    references_parser = _ReferencesParser()
else:
    references_parser = PydanticOutputParser(pydantic_object=References)
