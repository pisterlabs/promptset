from typing import Optional

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.llms.base import LLM
from langchain.sql_database import SQLDatabase
from pydantic import BaseModel, ConfigDict


class AppState(BaseModel):
    """Stores global state for the app.
    This class will be initialized during app startup, so most fields should be optional.
    """

    # SQLDatabase is not a pydantic model, we need to allow arbitrary types.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    warehouse: Optional[SQLDatabase] = None
    llm: Optional[LLM] = None
    coder_llm: Optional[LLM] = None
    toolkit: Optional[BaseToolkit] = None


app_state = AppState()
