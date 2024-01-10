from ast import literal_eval
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

import Globals
class TodoInput(BaseModel):
    """Input for Todo."""
    task: str = Field(
        ...,
        description="Description of the task"
    )

class TodoListInput(BaseModel):
    """Input for TodoList Generate."""
    todos: Optional[List[TodoInput]] = Field(
        default=None,
        # ...,
        description="List of tasks to add to or complete in the to-do list"
    )
    status: str = Field(
        ...,
        description="The operation on the to-do list (e.g., add, complete, check)"
    )

class TodoListTool(BaseTool):
    name = "todo_list_manager"
    description = f"""
    Manage a to-do list for users.
    Input should contain one of the following operations: , addcomplete and check.
    """
    global_todo_list: Dict[str, bool] = Globals.get_todo_list()
    #global_todo_list: Dict[str, bool] = {}

    def _run(self, status: str, todos: Optional[List[TodoInput]] = None):
        self.global_todo_list = Globals.get_todo_list()
        print("現在global_todo_list: ", self.global_todo_list)
        print("操作:", status)
        if todos:
            print("任務:", todos)
            if isinstance(todos, dict):
                todos = [todos]
        if status == "add":
            if todos:
                for todo in todos:
                    self.global_todo_list[todo['task']] = False
                print(self.get_todo_list())
                return self.get_todo_list()
            else: 
                return "未提供任務或任務不存在"
        elif status == "complete":
            if todos:
                for todo in todos:
                    if todo['task'] in self.global_todo_list:
                        self.global_todo_list[todo['task']] = True
                print(self.get_todo_list())
                return self.get_todo_list()
            else:
                return "未提供任務或任務不存在"
        elif status == "check":
            print(self.get_todo_list())
            return self.get_todo_list()
        else:
            return "請提供您要對TODO-List進行的操作"
        
    
    def get_todo_list(self):
        Globals.update_todo_list(self.global_todo_list)

        print("in get_todo_list Globals list: ", Globals.get_todo_list(), "self: ", self.global_todo_list)
        todo_list = [task for task, completed in self.global_todo_list.items() if not completed]
        completed_list = [task for task, completed in self.global_todo_list.items() if completed]

        return {
            "未完成任務": todo_list,
            "已完成任務": completed_list
        }

    args_schema: Optional[Type[BaseModel]] = TodoListInput
