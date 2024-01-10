from typing import Dict, Any, List, Optional
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler


class LLMInstrumentationHandler(BaseCallbackHandler):
    def on_agent_action(self, action: str, **kwargs) -> None:
        print(f"My custom handler, action: {action}")

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        print(
            f"My custom handler, prompts: {prompts}," f" \n {kwargs=} \n {serialized=}"
        )
