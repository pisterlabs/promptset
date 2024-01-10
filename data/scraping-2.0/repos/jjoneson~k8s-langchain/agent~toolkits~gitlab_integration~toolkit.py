from typing import List

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.callbacks.base import BaseCallbackManager
from langchain.tools import BaseTool

from tools.gitlab_integration.tool import GitlabChangeIssueLabelTool, GitlabCommentOnIssueTool, GitlabCommentOnMergeRequestTool, GitlabCreateIssueTool, GitlabCreateMergeRequestTool, GitlabListGroupsTool, GitlabListIssuesTool, GitlabListMergeRequestsTool, GitlabListProjectsTool, GitlabModel, GitlabReplyToIssueCommentTool, GitlabReplyToMergeRequestCommentTool

class GitlabIntegrationToolkit(BaseToolkit):
    """Toolkit for interacting with gitlab."""

    model: GitlabModel
    callback_manager: BaseCallbackManager | None
    
    class Config:
        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Return a list of tools."""
        return [
            GitlabListProjectsTool(model=self.model, callback_manager=self.callback_manager),
            GitlabListGroupsTool(model=self.model, callback_manager=self.callback_manager),
            GitlabListIssuesTool(model=self.model, callback_manager=self.callback_manager),
            GitlabChangeIssueLabelTool(model=self.model, callback_manager=self.callback_manager),
            GitlabListMergeRequestsTool(model=self.model, callback_manager=self.callback_manager),
            GitlabCreateIssueTool(model=self.model, callback_manager=self.callback_manager),
            GitlabCommentOnIssueTool(model=self.model, callback_manager=self.callback_manager),
            GitlabReplyToIssueCommentTool(model=self.model, callback_manager=self.callback_manager),
            GitlabCreateMergeRequestTool(model=self.model, callback_manager=self.callback_manager),
            GitlabCommentOnMergeRequestTool(model=self.model, callback_manager=self.callback_manager),
            GitlabReplyToMergeRequestCommentTool(model=self.model, callback_manager=self.callback_manager),
        ]
        