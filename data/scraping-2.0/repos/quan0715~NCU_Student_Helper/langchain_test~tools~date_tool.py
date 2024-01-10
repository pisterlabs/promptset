from datetime import datetime, date, time, timedelta
from typing import Any, Coroutine, Optional, Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class DateTimeInput(BaseModel):
    current_date: date = Field(
        description="Current date format as iso8601 which only contains date.")
    current_time: time = Field(
        description="Current time format as iso8601 which only contains time.")
    delta: timedelta = Field(
        description="The difference between target datetime and current datetime, format as iso8601 timedelta.")


class CurrentDateTimeTool(BaseTool):
    name = "CurrentDateTimeTool"
    description = "Useful to get the current date, time, and weekday."
    return_direct = False

    def _run(self, *args: Any, **kwargs: Any) -> str:
        return datetime.now().strftime("%c")

    def _arun(self, *args: Any, **kwargs: Any) -> Coroutine[Any, Any, Any]:
        raise Exception()


class DateTimeTool(BaseTool):
    name = "DateTimeTool"
    description = f"Useful to get desiring date, time, and weekday while current datetime is {datetime.now().strftime('%c')}"
    args_schema: Optional[Type[BaseModel]] = DateTimeInput
    return_direct = False

    def _run(self, current_date: date, current_time, delta: timedelta) -> str:
        return (datetime.combine(current_date, current_time) + delta).strftime("%c")

    def _arun(self, *args: Any, **kwargs: Any) -> Coroutine[Any, Any, Any]:
        raise Exception()
