from typing import List, Optional

from lanarky.responses import StreamingResponse
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain import PromptTemplate

from agent_backend.schemas import ModelSettings
from agent_backend.web.api.agent.agent_service.agent_service import AgentService
from agent_backend.web.api.agent.analysis import Analysis
from agent_backend.web.api.agent.helpers import (
    call_model_with_handling,
    parse_with_handling,
)
from agent_backend.web.api.agent.model_settings import create_model
from agent_backend.web.api.agent.prompts import (
    start_goal_prompt,
    analyze_task_prompt,
    create_tasks_prompt,
)
from agent_backend.web.api.agent.task_output_parser import TaskOutputParser
from agent_backend.web.api.agent.tools.tools import (
    get_tools_overview,
    get_tool_from_name,
    get_user_tools,
)

translator_prompt = PromptTemplate(
    template="""
     translate {text} to English.
    """,
    input_variables=["text"],
)


class OpenAIAgentService(AgentService):
    def __init__(self, model_settings: ModelSettings):
        self.model_settings = model_settings
        self._language = model_settings.language or "English"

    async def start_goal_agent(self, *, goal: str) -> List[str]:
        completion = await call_model_with_handling(
            self.model_settings,
            start_goal_prompt,
            {"goal": goal, "language": self._language},
        )

        task_output_parser = TaskOutputParser(completed_tasks=[])
        return parse_with_handling(task_output_parser, completion)
# 核心代码
    async def analyze_task_agent(
        self, *, goal: str, task: str, tool_names: List[str]
    ) -> Analysis:
        llm = create_model(self.model_settings)

        chain = LLMChain(llm=llm, prompt=analyze_task_prompt)
        pydantic_parser = PydanticOutputParser(pydantic_object=Analysis)
        tool_names = get_user_tools(tool_names)
        tools_overview = get_tools_overview(tool_names)
        print("可选工具：",tools_overview)
        trans_chain = LLMChain(llm=llm, prompt=translator_prompt)
        goal = trans_chain.run({"text": goal})
        task = trans_chain.run({"text": task})

        completion = chain.run(
            {
                "goal": goal,
                "task": task,
                "language": "English",
                "tools_overview": tools_overview
            }
        )
      
        try:
            return pydantic_parser.parse(completion)
        except Exception as error:
            return Analysis.get_default_analysis()
    #执行任务代理 
    async def execute_task_agent(
        self,
        *,
        goal: str,
        task: str,
        analysis: Analysis,
    ) -> StreamingResponse:
        tool_class = get_tool_from_name(analysis.action)
        print("断点：",tool_class,analysis)
        res = await tool_class(self.model_settings).call(goal, task, analysis.arg)
        return res

    async def create_tasks_agent(
        self,
        *,
        goal: str,
        tasks: List[str],
        last_task: str,
        result: str,
        completed_tasks: Optional[List[str]] = None,
    ) -> List[str]:
        llm = create_model(self.model_settings)
        chain = LLMChain(llm=llm, prompt=create_tasks_prompt)
        completion =  chain.run(
            {
                "goal": goal,
                "language": self._language,
                "tasks": tasks,
                "lastTask": last_task,
                "result": result,
            }
        )

        previous_tasks = (completed_tasks or []) + tasks
        task_output_parser = TaskOutputParser(completed_tasks=previous_tasks)
        return task_output_parser.parse(completion)
