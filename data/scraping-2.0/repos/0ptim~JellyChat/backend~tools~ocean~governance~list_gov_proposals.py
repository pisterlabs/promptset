from langchain.tools import StructuredTool
from pydantic import BaseModel, Field


from ..utils import getOcean


class ToolInputSchema(BaseModel):
    current: bool = Field(
        True,
        description="Set to false if you need all proposals. If set to true, only currenty live proposals will be returned.",
    )


def list_gov_proposal(current: bool) -> str:
    try:
        # Query Proposals
        status = "voting" if current else "all"
        proposals = []
        data = getOcean().governance.listGovProposals(status, "all", 0, True, size=200)
        proposals.extend(data.get("data"))

        next = data.get("page").get("next") if data.get("page") else None
        while next:
            data = getOcean().governance.listGovProposals(
                "all", "all", 0, True, size=200, next=next
            )
            proposals.extend(data.get("data"))
            next = data.get("page").get("next") if data.get("page") else None

        # Filter for Proposal ID and Title
        filtered_proposals = []
        for proposal in proposals:
            filtered_proposals.append(
                {"id": proposal.get("proposalId"), "title": proposal.get("title")}
            )
        return filtered_proposals
    except Exception as e:
        return str(e)


description = """Lists id and title of all proposals."""

governanceListGovProposalTool = StructuredTool(
    name="list_all_governance_proposal_ID_and_title",
    description=description,
    func=list_gov_proposal,
    args_schema=ToolInputSchema,
)
