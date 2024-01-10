import json

from langchain.schema import BaseOutputParser, OutputParserException
from langchain.chains.query_constructor.parser import get_parser
from typing import Any, Callable, Optional, Sequence
from langchain.chains.query_constructor.ir import (
    Comparator,
    Operator,
    StructuredQuery,
)


class StructuredQueryOutputParser(BaseOutputParser[StructuredQuery]):
    """Output parser that parses a structured query."""

    ast_parse: Callable

    def parse(self, text: Any) -> StructuredQuery:
        try:
            allowed_keys = ["query", "filter", "limit"]

            parsed = json.loads(text['text'])
            if len(parsed["query"]) == 0 or parsed["query"] == "NO_FILTER":
                parsed["query"] = " "

            if parsed["filter"] == "NO_FILTER" or not parsed["filter"]:
                parsed["filter"] = None
            else:
                parsed["filter"] = self.ast_parse(parsed["filter"])

            if not parsed.get("limit"):
                parsed.pop("limit", None)

            print('\n\nparsed:', parsed)
            return StructuredQuery(
                **{k: v for k, v in parsed.items() if k in allowed_keys}
            )
        except Exception as e:
            raise OutputParserException(
                f"Parsing text\n{text}\n raised following error:\n{e}"
            )

    @classmethod
    def from_components(
        cls,
        allowed_comparators: Optional[Sequence[Comparator]] = None,
        allowed_operators: Optional[Sequence[Operator]] = None,
    ) -> Any:
        ast_parser = get_parser(
            allowed_comparators=allowed_comparators, allowed_operators=allowed_operators
        )
        return cls(ast_parse=ast_parser.parse)
