from typing import List, Union, Optional, Type
from gentopia.model.param_model import OpenAIParamModel
from gentopia.agent.base_agent import BaseAgent
from gentopia.llm import OpenAIGPTClient
from gentopia.llm.base_llm import BaseLLM
from gentopia.model.agent_model import AgentType, AgentOutput
from gentopia.tools import BaseTool
from gentopia.utils.cost_helpers import calculate_cost
from pydantic import create_model, BaseModel

from gentpool.bench.grader.base import BaseGrader
from gentpool.bench.prompt import *


class ScoreGrader(BaseGrader):
    """
    A "score" Grader judge both the correctness and the quality of a prediction, and return a score from 0 to 100.
    """
    name: str = "ScoreGrader"
    type: AgentType = AgentType.vanilla
    grader_type = "score"
    version: str = ""
    description: str = "Grader agent judging the score of given prediciton. Input contains a task, a ground truth and a prediction. Output a score from 0 to 100."
    target_tasks: list[str] = []
    llm: BaseLLM = OpenAIGPTClient(model_name="gpt-4", params=OpenAIParamModel(temperature=0))
    prompt_template: PromptTemplate = TeacherStudentScorePrompt
    plugins: List[Union[BaseTool, BaseAgent]] = []
    examples: Union[str, List[str]] = None

    args_schema: Optional[Type[BaseModel]] = create_model("ScoreArgsSchema", task=(str, ...),
                                                          ground_truth=(str, ...),
                                                          prediction=(str, ...))

    def run(self, task: str, ground_truth: str, prediciton: str) -> AgentOutput:
        total_cost = 0
        total_token = 0

        prompt = self.prompt_template.format(task=task, ground_truth=ground_truth, prediction=prediciton)

        response = self.llm.completion(prompt)
        if response.state == "error":
            raise ValueError(f"{self.name} fails to retrieve response from LLM.")

        total_cost += calculate_cost(self.llm.model_name, response.prompt_token,
                                     response.completion_token)
        total_token += response.prompt_token + response.completion_token

        return AgentOutput(output=response.content, cost=total_cost, token_usage=total_token)

    def stream(self, *args, **kwargs) -> AgentOutput:
        raise NotImplementedError("ScoreGrader does not support streaming.")
