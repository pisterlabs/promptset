from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from ..utils import getOcean


class ToolInputSchema(BaseModel):
    placeholder: str = Field(..., description="Just fill in `asdf`")


def get_supply(placeholder: str) -> str:
    try:
        return getOcean().stats.getSupply()
    except Exception as e:
        return str(e)


description = """Gets supply information."""

statsGetSupplyTool = StructuredTool(
    name="gets_supply",
    description=description,
    func=get_supply,
    args_schema=ToolInputSchema,
)
