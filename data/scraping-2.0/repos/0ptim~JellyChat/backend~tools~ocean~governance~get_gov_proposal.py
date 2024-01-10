from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from ..utils import getOcean


class ToolInputSchema(BaseModel):
    txid: str = Field(..., description="The txid of the proposal")


def get_gov_proposal(txid: str) -> str:
    try:
        return getOcean().governance.getGovProposal(txid)
    except Exception as e:
        return str(e)


description = """Gets information about a proposal with given proposal id."""

governanceGetGovProposalTool = StructuredTool(
    name="get_governance_proposal",
    description=description,
    func=get_gov_proposal,
    args_schema=ToolInputSchema,
)
