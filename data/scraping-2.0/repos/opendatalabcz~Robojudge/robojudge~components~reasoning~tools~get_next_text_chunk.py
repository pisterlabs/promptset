from pydantic import BaseModel
from langchain.tools import StructuredTool


def get_text(texts: list[str]):
    for text in texts:
        yield text


SEARCH_TOOL_NAME = "get_next_text_chunk"


def create_get_text_tool(texts: list[str]):
    text_generator = get_text(texts)

    return StructuredTool.from_function(
        name=SEARCH_TOOL_NAME,
        func=lambda: next(text_generator),
        args_schema=GetTextSchema,
        description=f"Outputs the next chunk of the court ruling. Should be called repeatedly to get newer parts. Returns nothing if no parts are left. It contains {len(texts)} parts..",
        handle_tool_error=True,
    )


class GetTextSchema(BaseModel):
    ...
