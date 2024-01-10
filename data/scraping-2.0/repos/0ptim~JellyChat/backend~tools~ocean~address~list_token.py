from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from ..utils import getOcean


class ToolInputSchema(BaseModel):
    address: str = Field(..., description="The address")


def list_token(address: str) -> str:
    try:
        return getOcean().address.listToken(address)
    except Exception as e:
        return str(e)


description = """Gets the balance of all tokens on a address.
Contains: DFI, BTC, ETH, USDC, USDT, DOGE, DUSD, SPY, TSLA, APPL, ...
Does not contain DFI UTXO balance.
"""

addressListTokenTool = StructuredTool(
    name="get_token_balance",
    description=description,
    func=list_token,
    args_schema=ToolInputSchema,
)
