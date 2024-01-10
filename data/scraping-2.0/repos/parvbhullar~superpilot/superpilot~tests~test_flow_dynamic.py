import os
import platform
import sys
import time

import distro
import inflection
from pydantic import BaseModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from superpilot.core.ability.base import Ability
from typing import Dict, List
from typing_extensions import Literal
import asyncio
from superpilot.core.planning.schema import (
    LanguageModelClassification,
    LanguageModelResponse,
    Task,
)
from superpilot.core.pilot.task import SuperTaskPilot
from superpilot.core.configuration.config import get_config
from superpilot.core.planning.strategies.step_flow import StepFlow
from superpilot.core.planning.settings import LanguageModelConfiguration
from superpilot.core.resource.model_providers.openai import OpenAIModelName
from superpilot.core.resource.model_providers.schema import ModelProviderCredentials
from superpilot.framework.abilities import (
    TextSummarizeAbility,
)
from superpilot.core.resource.model_providers import (
    ModelProviderName,
    OpenAIProvider,
)
from superpilot.core.context.schema import Context
from superpilot.core.ability.super import SuperAbilityRegistry


class Step:
    def __init__(
        self,
        task: Task,
        abilities: List[Ability],
        execution_nature: Literal["single", "parallel", "seq"],
    ):
        self.task = task
        self.abilities = abilities
        self.execution_nature = execution_nature

    async def execute(self, **kwargs):
        if self.execution_nature == "parallel":
            tasks = [ability(**kwargs) for ability in self.abilities]
            res_list = await asyncio.gather(*tasks)
            for res in res_list:
                print(res.content)
        else:
            for ability in self.abilities:
                res = await ability(**kwargs)
                print(res.content)


class TaskPlanner:
    def __init__(self, flow: "Flow"):
        self.flow = flow
        self.tasks = []

    def define_task(self, task: Task):
        self.tasks.append(task)

    async def execute(self, **kwargs):
        print("Task plan:")
        for task in self.tasks:
            print(f" - {task.name}")

        confirmation = input("Confirm execution? (y/n): ")
        if confirmation.lower() == "y":
            for task in self.tasks:
                await task.execute(**kwargs)


class Flow:
    def __init__(self, steps: List[Step], strategy: Literal["prompt", "seq"] = "seq"):
        self.steps = steps
        self.strategy = strategy

    async def execute(self, **kwargs):
        if self.strategy == "prompt":
            await self._execute_with_prompt(**kwargs)
        else:
            await self._execute_sequential(**kwargs)

    async def _execute_with_prompt(self, **kwargs):
        for step in self.steps:
            print(f"Executing step: {step.name}")
            await step.execute(**kwargs)
            input("Press Enter to continue...")

    async def _execute_sequential(self, **kwargs):
        for step in self.steps:
            await step.execute(**kwargs)


class Ability(BaseModel):
    name: str
    arguments: Dict


class Step(BaseModel):
    step: str
    ability: Ability


class FlowModel(BaseModel):
    query: str
    steps: List[Step]


open_ai_creds = ModelProviderCredentials()
open_ai_creds.api_key = os.environ["OPENAI_API_KEY"]

OpenAIProvider.default_settings.credentials = ModelProviderCredentials()


class SimpleFlow:
    def __init__(self, query):
        self.query = query
        self.steps = []
        self.provider = OpenAIProvider()
        self.models = {
            LanguageModelClassification.FAST_MODEL: LanguageModelConfiguration(
                model_name=OpenAIModelName.GPT3,
                provider_name=ModelProviderName.OPENAI,
                temperature=0.9,
            ),
            LanguageModelClassification.SMART_MODEL: LanguageModelConfiguration(
                model_name=OpenAIModelName.GPT4,
                provider_name=ModelProviderName.OPENAI,
                temperature=0.9,
            ),
        }

    async def run(self, **kwargs):
        print(f"Executing query: {self.query}")
        prompt_strategy = StepFlow(**StepFlow.default_configuration.dict())
        template_kwargs = {}
        template_kwargs = self._make_template_kwargs_for_strategy(prompt_strategy)
        template_kwargs.update(kwargs)
        prompt = prompt_strategy.build_prompt(self.query, **template_kwargs)
        model_configuration = self.models[prompt_strategy.model_classification].dict()
        del model_configuration["provider_name"]
        response = await self.provider.create_language_completion(
            model_prompt=prompt.messages,
            functions=prompt.functions,
            **model_configuration,
            completion_parser=prompt_strategy.parse_response_content,
        )
        res = LanguageModelResponse.parse_obj(response.dict())
        flow = FlowModel(**res.content)
        return flow

    def _make_template_kwargs_for_strategy(self, strategy):
        template_kwargs = {
            "os_info": get_os_info(),
            "api_budget": self.provider.get_remaining_budget(),
            "current_time": time.strftime("%c"),
        }
        return template_kwargs


def get_os_info() -> str:
    os_name = platform.system()
    os_info = (
        platform.platform(terse=True)
        if os_name != "Linux"
        else distro.name(pretty=True)
    )
    return os_info


ALLOWED_ABILITY = {
    # SearchAndSummarizeAbility.name(): SearchAndSummarizeAbility.default_configuration,
    TextSummarizeAbility.name(): TextSummarizeAbility.default_configuration,
}
from superpilot.tests.test_env_simple import get_env
from superpilot.core.pilot import SuperPilot

# Flow executor -> Context
#
# Context -> Search & Summarise [WebSearch, KnowledgeBase, News] - parrallel] -> Context
# Context -> Analysis[FinancialAnalysis] -> Context
# Context -> Write[TextStream, PDF, Word] -> Context
# Context -> Respond[Twitter, Email, Stream]] -> Context
# Context -> Finalise[PDF, Word] -> Context

if __name__ == "__main__":
    user_objectives = "How to file the GSTR1"
    # user_objectives = "What is the weather in Mumbai"
    env = get_env({})

    planner = env.get("planning")
    ability_registry = env.get("ability_registry")

    pilot = SuperPilot(SuperPilot.default_settings, ability_registry, planner, env)
    asyncio.run(pilot.initialize(user_objectives))
    print(
        "***************** Pilot Initiated - Planing Started ******************************\n"
    )
    context_res = Context()
    model_providers = {ModelProviderName.OPENAI: OpenAIProvider()}
    config = get_config()
    ability_settings = SuperAbilityRegistry.default_settings
    # ability_settings.configuration.config = config
    ability_settings.configuration.model_providers = model_providers
    ability_settings.configuration.abilities = {
        ability_name: ability for ability_name, ability in ALLOWED_ABILITY.items()
    }
    super_ability_registry = SuperAbilityRegistry(
        ability_settings,
        environment=env,
    )
    search_step = SuperTaskPilot(super_ability_registry, model_providers)
    flow = SimpleFlow(user_objectives)
    print("\n\n")
    t1 = time.time()
    print(f"******************** Runing Flow ********************\n")
    flow_res = asyncio.run(
        flow.run(ability_schema=super_ability_registry.dump_abilities())
    )
    print(f"****************** Executed Flow******************\n")
    t2 = time.time()
    print(f"Time Taken: {round(t2-t1, 4)} Secs")
    print("\n\n")
    print("***************** Flow Result ******************************\n")
    print(flow_res.dict())
    print("\n***************** Flow End ******************************")
    if flow_res and flow_res.steps:
        for index, step in enumerate(flow_res.steps):
            objective = step.ability.arguments.pop("query", None)
            if not objective:
                continue
            task = Task(
                objective=objective,
                priority=1,
                type="text",
                ready_criteria=[],
                acceptance_criteria=[],
            )
            kargs = {
                "query": objective,
                "task_context": task.context,
                **step.ability.arguments,
            }
            ability_settings = SuperAbilityRegistry.default_settings
            # ability_settings.configuration.config = config

            abilities = {}
            abilities[step.ability.name] = ALLOWED_ABILITY[step.ability.name]
            ability_settings.configuration.abilities = abilities

            super_ability_registry = SuperAbilityRegistry(
                ability_settings,
                environment=env,
            )
            print("\n\n")
            t1 = time.time()
            print(f"******************** Runing Goal {index+1} ********************\n")
            print(f"Objective: {objective}")
            print(f"abilities: {abilities}")
            search_step = SuperTaskPilot(super_ability_registry, model_providers)
            context_res = asyncio.run(search_step.execute(task, context_res, **kargs))
            print(f"****************** Executed Goal {index+1} ******************\n")
            t2 = time.time()
            print(f"Time Taken: {round(t2-t1, 4)} Secs")

    print("\n\n\n\n")
    print("************************************************************")
    print(context_res.format_numbered())
    file_name = inflection.underscore(user_objectives).replace(" ", "_")
    file_name = f"{file_name}.txt"
    context_res.to_file(file_name)
    print(f"File Saved: {file_name}")
    print("************************************************************")
