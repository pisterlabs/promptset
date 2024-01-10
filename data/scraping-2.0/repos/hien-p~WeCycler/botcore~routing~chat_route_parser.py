from typing import Dict, Any
from langchain.schema import OutputParserException
from langchain.chains.router.llm_router import RouterOutputParser

class ChatRouterOutputParser(RouterOutputParser):
    """Parser for output of router chain int he multi-prompt chain."""

    default_destination: str = "DEFAULT"
    
    def parse(self, text: str) -> Dict[str, Any]:
        try:
            parsed = super().parse(text)
            parsed["next_inputs"]['question'] = parsed["question"]
            return parsed
        except Exception as e:
            raise OutputParserException(
                f"Parsing text\n{text}\n raised following error:\n{e}"
            )    
