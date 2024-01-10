from typing import Any, List, Optional, Type

from pydantic import BaseModel, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from bespokebots.services.agent.todoist_tools.base import TodoistBaseTool
from bespokebots.models.tools.todoist import (
    TodoistProject, 
    TodoistTask,
    TodoistDue
)

from todoist_api_python.models import Project


class GetProjectIdsSchema(BaseModel):
    """Schema for the GetProjectIdsTool."""
    
    project_names: List[str] = Field(
        ...,
        title="Project Names",
        description="The names of the projects to retrieve the IDs of. An empty list will return all project IDs."
    )

class GetProjectIdsTool(TodoistBaseTool):
    name: str = "get_project_ids"
    description: str = """Use this tool when you need to find one or more project_ids based on the user's request.  This tool just returns a mapping of project name to project id, which results in far fewer tokens being used.  this tool is more efficient than the view_projects tool, which returns the full project, including its active tasks."""

    args_schema: Type[GetProjectIdsSchema] = GetProjectIdsSchema

    def _run(
        self,   
        project_names: List[str],
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> dict:
        """View projects in Todoist.
        Returns a dictionary with the following structure:
        {
            [{project_name: "name", project_id: "id"}]
        }
        """
        try:
            todoist_projects = self.todoist_client.get_projects()
            if len(project_names) > 0:
                project_mappings =[ {'project_name': p.name, 'project_id': p.id} for p in todoist_projects if p.name in project_names] 
            else:
                project_mappings =[ {'project_name': p.name, 'project_id': p.id} for p in todoist_projects]

            return project_mappings
        except Exception as e:
            raise e

    async def _arun(self, *args: Any, **kwargs: Any) -> dict:
        raise NotImplementedError(f"The tool {self.name} does not support async yet.")

    