from __future__ import annotations
from typing import Any, List, Optional, Type
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
import logging

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from bespokebots.services.agent.todoist_tools.base import TodoistBaseTool
from bespokebots.models.tools.todoist import TodoistProject, TodoistTask, TodoistDue

from todoist_api_python.models import Project


logging.basicConfig(level=logging.INFO)
# Initialize the logger
logger = logging.getLogger(__name__)


class FindTasksWithFilterSchema(BaseModel):
    """Schema for the FindTasksWithFilterTool."""

    filter: str = Field(
        ...,
        title="Filter",
        description="The filter to use to find tasks. For example, to find all the tasks in the Personal project which are due today, you would set the filter parameter to 'today & #Personal'.",
    )


class FindTasksWithFilterTool(TodoistBaseTool):
    name: str = "find_tasks_with_filter"
    description: str = """Use this tool to find tasks in Todoist using a filter. You can use any filter that you can use in Todoist. Do not use the a project's project_id with this filter, prefix the project's name with the # symbol instead. For example, to find all the tasks in the Personal project which are due today, you would set the filter parameter to 'today & #Personal'. You can perform keyword searches by prefixing the keyword(s) with the term "search:".  For example, if you wanted to find a task named "feed the cats" in a Chores project, your filter query would be 'search: feed the cats & #Chores'. Todoist follows a regular pattern for boolean symbols. Additionally, sections can be specified with the / character.  For example, to find all 'InProgress' tasks in the Personal project, you would set the filter parameter to '#Personal & /InProgress'."""

    args_schema: Type[FindTasksWithFilterSchema] = FindTasksWithFilterSchema

    def _run(
        self, filter: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> dict:
        try:
            tasks = self.todoist_client.get_tasks(filter=filter)
            if tasks:
                the_tasks = [TodoistTask.from_todoist(task).dict() for task in tasks]
                return {"tasks": the_tasks}
            else:
                logger.info(f"No tasks were found which match the filter: {filter}")
                return {
                    "tasks": f"No tasks were found which match the filter: {filter}"
                }

        except Exception as e:
            logger.exception(f"Error running the %s tool: %s", self.name, e)
            raise e

    async def _arun(
        self,
        project_names: Optional[List[str]],
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> dict:
        raise NotImplementedError(f"The tool {self.name} does not support async yet.")
