import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from superpilot.core.ability import (
    AbilityAction,
    AbilityRegistry,
    SimpleAbilityRegistry,
)
from superpilot.core.pilot.base import Pilot
from superpilot.core.pilot.settings import (
    PilotSettings,
    PilotSystemSettings,
    PilotConfiguration,
    PilotSystems,
    ExecutionAlgo
)
from superpilot.core.configuration import Configurable
from superpilot.core.memory import SimpleMemory
from superpilot.core.environment import SimpleEnv
from superpilot.core.planning import SimplePlanner, Task, TaskStatus
from superpilot.core.plugin.simple import (
    PluginLocation,
    PluginStorageFormat,
    SimplePluginService,
)
from superpilot.core.resource.model_providers import OpenAIProvider
from superpilot.core.workspace.simple import SimpleWorkspace


class SuperPilot(Pilot, Configurable):

    default_settings = PilotSystemSettings(
        name="super_pilot",
        description="A super pilot.",
        configuration=PilotConfiguration(
            name="Entrepreneur-GPT",
            role=(
                "An AI designed to autonomously develop and run businesses with "
                "the sole goal of increasing your net worth."
            ),
            goals=[
                "Increase net worth",
                "Grow Twitter Account",
                "Develop and manage multiple businesses autonomously",
            ],
            cycle_count=0,
            max_task_cycle_count=3,
            creation_time="",
            execution_algo=ExecutionAlgo.PLAN_AND_EXECUTE,
        ),
    )

    def __init__(
        self,
        settings: PilotSystemSettings,
        ability_registry: AbilityRegistry,
        planning: SimplePlanner,
        environment: SimpleEnv,
    ):
        self._configuration = settings.configuration
        self._logger = environment.get("logger")
        self._ability_registry = ability_registry
        self._memory = environment.get("memory")
        # FIXME: Need some work to make this work as a dict of providers
        #  Getting the construction of the config to work is a bit tricky
        self._openai_provider = environment.get("model_providers")["openai"]
        self._planning = planning
        self._workspace = environment.get("workspace")
        self._task_queue = []
        self._completed_tasks = []
        self._current_task = None
        self._next_step = None

    async def initialize(
            self,
            user_objective: str,
            *args, **kwargs
    ) -> dict:
        self._logger.debug("Initializing SuperPilot.")
        model_response = await self._planning.decide_name_and_goals(
            user_objective,
        )
        self._logger.debug(f"Model response: {model_response.content}")
        self._configuration.name = model_response.content["pilot_name"]
        self._configuration.role = model_response.content["pilot_role"]
        self._configuration.goals = model_response.content["pilot_goals"]
        self._configuration.creation_time = datetime.now().isoformat()
        return model_response.content

    async def plan(self):
        return await self.build_initial_plan()

    async def execute(self, *args, **kwargs):
        self._logger.info(f"Executing step {self._configuration.cycle_count}")
        pass

    async def watch(self, *args, **kwargs):

        pass

    async def build_initial_plan(self) -> dict:
        # TODO: split query into mulitple queries to answer the question using the planner
        plan = await self._planning.make_initial_plan(
            pilot_name=self._configuration.name,
            pilot_role=self._configuration.role,
            pilot_goals=self._configuration.goals,
            abilities=self._ability_registry.list_abilities(),
        )
        tasks = [Task.parse_obj(task) for task in plan.content["task_list"]]

        # TODO: Should probably do a step to evaluate the quality of the generated tasks,
        #  and ensure that they have actionable ready and acceptance criteria

        self._task_queue.extend(tasks)
        self._task_queue.sort(key=lambda t: t.priority, reverse=True)
        self._task_queue[-1].context.status = TaskStatus.READY
        return plan.content

    async def determine_next_step(self, *args, **kwargs):
        if not self._task_queue:
            return {"response": "I don't have any tasks to work on right now."}

        self._configuration.cycle_count += 1
        task = self._task_queue.pop()
        self._logger.info(f"Working on task: {task}")

        task = await self._evaluate_task_and_add_context(task)
        next_ability = await self._choose_next_step(
            task,
            self._ability_registry.dump_abilities(),
        )
        self._current_task = task
        self._next_step = next_ability.content
        return self._current_task, self._next_step

    async def execute_next_step(self, user_input: str, *args, **kwargs):
        if user_input == "y":
            ability = self._ability_registry.get_ability(
                self._next_step["next_ability"]
            )
            ability_response = await ability(**self._next_step["ability_arguments"])
            await self._update_tasks_and_memory(ability_response)
            if self._current_task.context.status == TaskStatus.DONE:
                self._completed_tasks.append(self._current_task)
            else:
                self._task_queue.append(self._current_task)
            self._current_task = None
            self._next_step = None

            return ability_response.dict()
        else:
            raise NotImplementedError

    async def _evaluate_task_and_add_context(self, task: Task) -> Task:
        """Evaluate the task and add context to it."""
        if task.context.status == TaskStatus.IN_PROGRESS:
            # Nothing to do here
            return task
        else:
            self._logger.debug(f"Evaluating task {task} and adding relevant context.")
            # TODO: Look up relevant memories (need working memory system)
            # TODO: Evaluate whether there is enough information to start the task (language model call).
            task.context.enough_info = True
            task.context.status = TaskStatus.IN_PROGRESS
            return task

    async def _choose_next_step(self, task: Task, ability_schema: list[dict]):
        """Choose the next ability to use for the task."""
        self._logger.debug(f"Choosing next ability for task {task}.")
        if task.context.cycle_count > self._configuration.max_task_cycle_count:
            # Don't hit the LLM, just set the next action as "breakdown_task" with an appropriate reason
            raise NotImplementedError
        elif not task.context.enough_info:
            # Don't ask the LLM, just set the next action as "breakdown_task" with an appropriate reason
            raise NotImplementedError
        else:
            next_ability = await self._planning.determine_next_step(
                task, ability_schema
            )
            return next_ability

    async def _update_tasks_and_memory(self, ability_result: AbilityAction):
        self._current_task.context.cycle_count += 1
        self._current_task.context.prior_actions.append(ability_result)
        # TODO: Summarize new knowledge
        # TODO: store knowledge and summaries in memory and in relevant tasks
        # TODO: evaluate whether the task is complete

    def __repr__(self):
        return "SuperPilot()"


def _prune_empty_dicts(d: dict) -> dict:
    """
    Prune branches from a nested dictionary if the branch only contains empty dictionaries at the leaves.

    Args:
        d: The dictionary to prune.

    Returns:
        The pruned dictionary.
    """
    pruned = {}
    for key, value in d.items():
        if isinstance(value, dict):
            pruned_value = _prune_empty_dicts(value)
            if (
                pruned_value
            ):  # if the pruned dictionary is not empty, add it to the result
                pruned[key] = pruned_value
        else:
            pruned[key] = value
    return pruned
