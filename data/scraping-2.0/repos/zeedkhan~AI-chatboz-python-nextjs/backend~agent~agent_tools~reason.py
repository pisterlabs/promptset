from backend.agent.tool import Tool
from langchain import LLMChain
from typing import Any


class Reason(Tool):
    description = (
        "Reason about task via existing information or understanding. "
        "Make decisions / selections from options."
    )

    async def call(
        self, goal: str, task: str, input_str: str
    ) -> Any:
        from backend.agent.prompt import execute_task_prompt

        chain = LLMChain(llm=self.model, prompt=execute_task_prompt)
# StreamingResponse.from_chain(
#             chain,
#             {"goal": goal, "language": self.language, "task": task},
#             media_type="text/event-stream",
        # )
        return await chain.arun(
            {"goal": goal, "language": self.language, "task": task}
        )
