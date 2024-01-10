"""Diet Plan Toolkit Object

Toolkit object for diet plan related tools

"""
from typing import List, Optional
from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools import BaseTool

from healthfirstai_prototype.models.data_models import User
from .tools import (
    WorkoutScheduleTool,
    EditWorkoutScheduleTool,
)


class ExerciseToolkit(BaseToolkit):
    """Exercise Toolkit Object

    Parameters:
        user_info (User): User information

    Methods:
        get_tools: Get the tools in the toolkit
    """

    user_info: Optional[User] = None

    # related tasks
    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            WorkoutScheduleTool(),
            EditWorkoutScheduleTool(),
        ]
