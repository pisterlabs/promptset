from langchain.callbacks.base import BaseCallbackHandler

from typing import Optional, Any, Union, Dict
from uuid import UUID

from langchain.schema import (
    LLMResult,
)

from langchain.schema import (
    AgentAction,
    AgentFinish
)

class StreamlitAgentCallbackHandler(BaseCallbackHandler):

    def __init__(self, st) -> None:
        # super().__init__()
        self.st = st

    def write(self, message):
        self.st.write(message)
    
    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on agent end."""
        try:
            self.write(f"> {finish.log}")
        except Exception as e:
            print("Could not execute on agent finish.", e)


    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool ends running."""
        try:
            self.write(f"on_tool_end: {output}")
        except Exception as e:
            print("Could not execute on agent on_tool_end.", e)


        
