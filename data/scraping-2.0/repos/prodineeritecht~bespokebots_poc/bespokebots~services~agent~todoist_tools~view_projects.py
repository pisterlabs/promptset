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


class ViewProjectsSchema(BaseModel):
    """Schema for the ViewProjectsTool."""
    
    project_id: Optional[str] = Field(
       None,
       title="Project ID",
       description="The ID of the project to view."
    )

    project_names: Optional[List[str]] = Field(
        None,
        title="Project Names",
        description="The names of the project to retrieve a summary off"
    )

class ViewProjectsTool(TodoistBaseTool):

    name: str = "view_projects"
    description: str = """Use this tool to help you answer questions a human may have about their projects and tasks in their task management system. If the human doesn't specify a project, this tool will return all the human's projects and each project's open tasks. If the human does specify a project, this tool will return that specific project and all its open tasks.  You may ask for multiple projects by name as well."""

    args_schema: Type[ViewProjectsSchema] = ViewProjectsSchema

    def _run(
        self,   
        project_id: Optional[str] = None,
        project_names: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> dict:
        """View projects in Todoist."""
        try:
            if project_id:
                todoist_projects =[self.todoist_client.get_project(project_id=project_id)]
            else:
                todoist_projects = self.todoist_client.get_projects()

            if project_names and len(project_names) > 0 and len(todoist_projects) > 1:
                todoist_projects = [p for p in todoist_projects if p.name in project_names]
            
            projects = []
            for project in todoist_projects:
                #transform the project to a bespoke bots type
                bb_project = TodoistProject.from_todoist(project)
                #retrieve all the open tasks for this project
                todoist_tasks = self.todoist_client.get_tasks(project_id=project.id)
                for task in todoist_tasks:
                    bb_task = TodoistTask.from_todoist(task)
                    bb_project.tasks.append(bb_task)
                projects.append(bb_project)
            return TodoistProject.project_list_to_dict(projects)
        except Exception as e:
            raise Exception(f"An error occurred: {e}")

    async def _arun(self, project_names: Optional[List[str]], run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> dict:
        raise NotImplementedError(f"The tool {self.name} does not support async yet.")
