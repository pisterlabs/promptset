from typing import Any, Dict, List, cast
from langchain.tools import BaseTool
from policies.available_policies import UnknownPolicies
from policies.ignore import IgnorePolicy

from tools.versioned_vector_store import VersionedVectorStoreTool


class UnknownTool(BaseTool):
    """A tool to use when the agent doesn't know about an answer.

    This takes in as input the last tool that the agent used
    along with the intermediate steps. It then performs an action
    based on the unknown policy of the tool, using the intermediate
    steps if needed.
    """

    def _run(self, last_tool: BaseTool, intermediate_steps: Dict[str, Any]) -> Any:
        """Use the tool.

        Args:
            last_tool: The last tool that the agent used.
            intermediate_steps: The intermediate steps from the agent execution.


        Returns:
            The answer to the question.
        """
        # if tool is not of class VersionedVectorStoreTool, assume that the
        # unknown policy for it is Ignore.
        if not isinstance(last_tool, VersionedVectorStoreTool):
            policy = IgnorePolicy()
        else:
            policy = last_tool.unknown_policy

        # implement the policy
        prompt = policy.implement(intermediate_steps=intermediate_steps)

        # return the prompt
        return prompt
