import json
import os
import random
import time
from typing import Tuple, Dict, List

from gentopia.agent import BaseAgent
from gentopia.llm import OpenAIGPTClient
from gentopia.model import AgentOutput
from gentopia.output.base_output import BaseOutput

from gentpool.bench.eval import BaseEval
from gentpool.bench.eval.base_eval import EvalResult
from gentpool.bench.grader import *


class QAEval(BaseEval):
    """
    Evaluation class for QA tasks. 
    Such tasks should have the following keys in the json file:
    - problem: the problem description
    - solution: the solution to the problem
    """
    eval_class: str
    eval_subclass: str
    grader: BaseGrader = GateGrader(llm=OpenAIGPTClient(model_name="gpt-4"))

    def evaluate(self, agent: BaseAgent, n_smaple: int, seed=0, private=False, verbose=True) -> Tuple[
        EvalResult, List[Tuple[List[str], Dict]]]:
        Log = []
        result = EvalResult()
        for index in range(n_smaple):
            _, index, eval_result, response, eval_log = self.evaluate_single(agent, 0, n_smaple, seed, private)
            _, grade_result, grade_log = self.grade_single(response, index)
            result += eval_result + grade_result
            Log.append((eval_log, grade_log))

        return result.avg(n_smaple), Log

    def eval_async(self, agent: BaseAgent, n_smaple: int, seed: int = 0, *args, **kwargs) -> EvalResult:
        raise NotImplementedError("Async evaluation not supported yet.")

    def evaluate_single(self, agent: BaseAgent, index: int, n_smaple: int, seed=0, private=False) \
            -> Tuple["QAEval", int, EvalResult, AgentOutput, List[str]]:
        if self.data is None:
            self.data = self._get_data(seed, private, n_smaple)

        result = EvalResult(avg_runtime=time.time())
        task = self.data[index]
        opt = BaseOutput()
        agent_instruction = task.get("problem", None)
        agent_log = []
        try:
            response = agent.run(agent_instruction, opt)
            if hasattr(agent, "message_scratchpad"):
                agent_log = agent.message_scratchpad[-1]
            assert response is not None
        except Exception as e:
            result.fail_rate = 1
            response = AgentOutput(output="Agent failed", cost=0, token_usage=0)
        result.avg_runtime = time.time() - result.avg_runtime
        result.avg_cost = response.cost
        result.avg_token_usage = response.token_usage
        return self, index, result, response, agent_log + opt.log

    def grade_single(self, response: AgentOutput, index:int) -> Tuple["QAEval", EvalResult, Dict]:
        task = self.data[index]
        result = EvalResult()
        agent_instruction = task.get("problem", None)
        if response.output != "Agent failed":
            try:
                grader_output = self.grader.run(task=agent_instruction,
                                                ground_truth=task["solution"],
                                                prediciton=response.output)
            except Exception as e:
                grader_output = AgentOutput(output="Grader failed", cost=0, token_usage=0)
            result.eval_cost = grader_output.cost
            if isinstance(self.grader, GateGrader):
                result.score = 1 if "pass" in grader_output.output.lower() else 0
            elif isinstance(self.grader, ScoreGrader):
                try:
                    result.score = int(grader_output.output) / 100
                except Exception as e:
                    result.score = 0
            else:
                raise NotImplementedError("Grader type not supported.")

        return self, result, \
            dict(
                prompt = self.grader.prompt_template.format(task=agent_instruction, ground_truth=task["solution"], prediction=response.output),
                output = grader_output.output,
                solution = task["solution"]
            )
