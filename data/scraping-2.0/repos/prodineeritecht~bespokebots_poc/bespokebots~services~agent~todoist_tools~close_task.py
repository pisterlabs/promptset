from __future__ import annotations
from typing import Any, List, Optional, Type
from typing import TYPE_CHECKING

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


class CloseTaskSchema(BaseModel):

    task_id: str = Field(
        ...,
        title="Task ID",
        description="The ID of the task to close."
    )


class CloseTaskTool(TodoistBaseTool):
    
    name: str = "close_task"
    description: str = """Use this tool to close a task in Todoist."""

    args_schema: Type[CloseTaskSchema] = CloseTaskSchema

    def _run(
        self,   
        task_id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> dict:
        """Close a task in Todoist."""
        try:
            task = self.todoist_client.close_task(task_id=task_id)
            return {"task_id": task_id}
        except Exception as e:
            raise Exception(f"An error occurred when trying to delete a task: {e}")
            

    async def _arun(self, *args: Any, **kwargs: Any) -> dict:
            raise NotImplementedError(f"The tool {self.name} does not support async yet.")