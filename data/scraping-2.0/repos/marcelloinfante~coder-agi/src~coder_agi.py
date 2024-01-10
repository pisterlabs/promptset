from src.openai_model import OpenAIModel
from src.initializer import Initializer
from src.controller import Controller
from src.tools.enum import ToolsEnum
from src.evaluator import Evaluator
from src.executor import Executor
from src.tools.index import Tool
from src.planner import Planner
from src.config import Config
from src.memory import Memory
from src.logger import Logger
from src.vector_db import VectorDB

from src.planner_tot import PlannerTOT


class CoderAGI:
    def __init__(self, config_path):
        self.config = Config(config_path)
        self.memory = Memory(self.config)
        self.llm = OpenAIModel(self.config)
        self.vector_db = VectorDB(self.config, self.llm, self.memory)

    def run(self):
        # planner = PlannerTOT(self.llm, self.config, self.memory)
        # planner.plan()
        Initializer.initialize(self.config)

        for _ in range(self.config.max_steps):
            Planner.plan(self.llm, self.config, self.memory)
            Logger.log_plan(self.memory)

            current_objective = self.memory.plan.pop(0)

            while True:
                query_response = self.vector_db.query(current_objective)

                executor_response = Executor.execute(
                    self.llm,
                    self.memory,
                    self.config,
                    current_objective,
                    query_response,
                )

                Logger.log_executor_response(executor_response)

                tool_response = Tool.execute(
                    executor_response,
                    self.config,
                    self.llm,
                    query_response,
                )

                Logger.log_tool_response(tool_response)

                evaluator_response = Evaluator.evaluate(
                    self.llm,
                    self.config,
                    executor_response,
                    tool_response,
                    current_objective,
                )

                Logger.log_evaluation_response(evaluator_response)

                Controller.control(
                    self.memory,
                    self.config,
                    current_objective,
                    executor_response,
                    tool_response,
                    evaluator_response,
                )

                # self.vector_db.update_collection()

                if evaluator_response["is_finish"]:
                    objective = {
                        "title": current_objective["title"],
                        "description": current_objective["description"],
                    }

                    self.memory.objetives.append(objective)
                    self.memory.reset_states()

                    break

                if self.memory.errors == 5:
                    objective = {
                        "title": current_objective["title"],
                        "description": current_objective["description"],
                    }

                    self.memory.error_objetives.append(objective)
                    self.memory.make_plan = True
                    self.memory.reset_states()

                    break
