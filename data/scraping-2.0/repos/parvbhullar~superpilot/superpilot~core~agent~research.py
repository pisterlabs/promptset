import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel

from superpilot.core.ability import (
    AbilityRegistrySettings,
    AbilityAction,
    ResearchAbilityRegistry,
)
from superpilot.core.pilot.base import Agent
from superpilot.core.configuration import (
    Configurable,
    SystemConfiguration,
    SystemSettings,
    WorkspaceSettings,
)
from superpilot.core.memory import MemorySettings, SimpleMemory
from superpilot.core.planning import PlannerSettings, Task, TaskStatus, ResearchPlanner
from superpilot.core.plugin.research import (
    ResearchPluginService,
    PluginLocation,
    PluginStorageFormat,
)
from superpilot.core.resource.model_providers import OpenAIProvider, OpenAISettings
from superpilot.core.workspace.research import ResearchWorkspace


class AgentSystems(SystemConfiguration):
    ability_registry: PluginLocation
    memory: PluginLocation
    openai_provider: PluginLocation
    planning: PluginLocation
    workspace: PluginLocation


class AgentConfiguration(SystemConfiguration):
    cycle_count: int
    max_task_cycle_count: int
    creation_time: str
    name: str
    role: str
    goals: list[str]
    systems: AgentSystems


class AgentSystemSettings(SystemSettings):
    configuration: AgentConfiguration


class AgentSettings(BaseModel):
    pilot: AgentSystemSettings
    ability_registry: AbilityRegistrySettings
    memory: MemorySettings
    openai_provider: OpenAISettings
    planning: PlannerSettings
    workspace: WorkspaceSettings

    def update_pilot_name_and_goals(self, pilot_goals: dict) -> None:
        self.pilot.configuration.name = pilot_goals["pilot_name"]
        self.pilot.configuration.role = pilot_goals["pilot_role"]
        self.pilot.configuration.goals = pilot_goals["pilot_goals"]


class ResearchAgent(Agent, Configurable):
    default_settings = AgentSystemSettings(
        name="research_pilot",
        description="A research pilot.",
        configuration=AgentConfiguration(
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
            systems=AgentSystems(
                ability_registry=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="superpilot.core.ability.ResearchAbilityRegistry",
                ),
                memory=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="superpilot.core.memory.SimpleMemory",
                ),
                openai_provider=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="superpilot.core.resource.model_providers.OpenAIProvider",
                ),
                planning=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="superpilot.core.planning.ResearchPlanner",
                ),
                workspace=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="superpilot.core.workspace.ResearchWorkspace",
                ),
            ),
        ),
    )

    def __init__(
        self,
        settings: AgentSystemSettings,
        logger: logging.Logger,
        ability_registry: ResearchAbilityRegistry,
        memory: SimpleMemory,
        openai_provider: OpenAIProvider,
        planning: ResearchPlanner,
        workspace: ResearchWorkspace,
    ):
        self._configuration = settings.configuration
        self._logger = logger
        self._ability_registry = ability_registry
        self._memory = memory
        # FIXME: Need some work to make this work as a dict of providers
        #  Getting the construction of the config to work is a bit tricky
        self._openai_provider = openai_provider
        self._planning = planning
        self._workspace = workspace
        self._task_queue = []
        self._completed_tasks = []
        self._current_task = None
        self._next_ability = None

    @classmethod
    def load_pilot(
        cls,
        pilot_settings: Dict,
        logger: logging.Logger,
    ) -> "ResearchAgent":
        pilot_args = {}

        pilot_args["settings"] = pilot_settings.pilot
        pilot_args["logger"] = logger
        pilot_args["workspace"] = cls._get_system_instance(
            "workspace",
            pilot_settings,
            logger,
        )
        pilot_args["openai_provider"] = cls._get_system_instance(
            "openai_provider",
            pilot_settings,
            logger,
        )
        pilot_args["planning"] = cls._get_system_instance(
            "planning",
            pilot_settings,
            logger,
            model_providers={"openai": pilot_args["openai_provider"]},
        )
        pilot_args["memory"] = cls._get_system_instance(
            "memory",
            pilot_settings,
            logger,
            workspace=pilot_args["workspace"],
        )

        pilot_args["ability_registry"] = cls._get_system_instance(
            "ability_registry",
            pilot_settings,
            logger,
            workspace=pilot_args["workspace"],
            memory=pilot_args["memory"],
            model_providers={"openai": pilot_args["openai_provider"]},
        )

        return cls(**pilot_args)

    @classmethod
    def from_workspace(
        cls,
        workspace_path: Path,
        logger: logging.Logger,
    ) -> "ResearchAgent":
        pilot_settings = ResearchWorkspace.load_pilot_settings(
            workspace_path, cls.name()
        )
        return cls.load_pilot(pilot_settings, logger)

    @classmethod
    def from_goal(
        cls,
        goal: str,
        thread_id: str,
        logger: logging.Logger,
    ) -> "ResearchAgent":
        pilot_settings = ResearchWorkspace.load_pilot_settings_goal(
            goal, thread_id, cls.name()
        )
        return cls.load_pilot(pilot_settings, logger)

    async def build_initial_plan(self) -> dict:
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

    async def determine_next_ability(self, *args, **kwargs):
        if not self._task_queue:
            return {"response": "I don't have any tasks to work on right now."}

        self._configuration.cycle_count += 1
        task = self._task_queue.pop()
        self._logger.info(f"Working on task: {task}")

        task = await self._evaluate_task_and_add_context(task)
        next_ability = await self._choose_next_ability(
            task,
            self._ability_registry.dump_abilities(),
        )
        self._current_task = task
        self._next_ability = next_ability.content
        return self._current_task, self._next_ability

    async def execute_next_ability(self, user_input: str, *args, **kwargs):
        if user_input == "y":
            ability = self._ability_registry.get_ability(
                self._next_ability["next_ability"]
            )
            ability_response = await ability(**self._next_ability["ability_arguments"])
            await self._update_tasks_and_memory(ability_response)
            if self._current_task.context.status == TaskStatus.DONE:
                self._completed_tasks.append(self._current_task)
            else:
                self._task_queue.append(self._current_task)
            self._current_task = None
            self._next_ability = None

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

    async def _choose_next_ability(self, task: Task, ability_schema: list[dict]):
        """Choose the next ability to use for the task."""
        self._logger.debug(f"Choosing next ability for task {task}.")
        if task.context.cycle_count > self._configuration.max_task_cycle_count:
            # Don't hit the LLM, just set the next action as "breakdown_task" with an appropriate reason
            raise NotImplementedError
        elif not task.context.enough_info:
            # Don't ask the LLM, just set the next action as "breakdown_task" with an appropriate reason
            raise NotImplementedError
        else:
            next_ability = await self._planning.determine_next_ability(
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
        return "ResearchAgent()"

    ################################################################
    # Factory interface for pilot bootstrapping and initialization #
    ################################################################

    @classmethod
    def build_user_configuration(cls) -> dict[str, Any]:
        """Build the user's configuration."""
        configuration_dict = {
            "pilot": cls.get_user_config(),
        }

        system_locations = configuration_dict["pilot"]["configuration"]["systems"]
        for system_name, system_location in system_locations.items():
            system_class = ResearchPluginService.get_plugin(system_location)
            configuration_dict[system_name] = system_class.get_user_config()
        configuration_dict = _prune_empty_dicts(configuration_dict)
        return configuration_dict

    @classmethod
    def compile_settings(
        cls, logger: logging.Logger, user_configuration: dict
    ) -> AgentSettings:
        """Compile the user's configuration with the defaults."""
        logger.debug("Processing pilot system configuration.")
        configuration_dict = {
            "pilot": cls.build_pilot_configuration(
                user_configuration.get("pilot", {})
            ).dict(),
        }

        system_locations = configuration_dict["pilot"]["configuration"]["systems"]

        # Build up default configuration
        for system_name, system_location in system_locations.items():
            logger.debug(f"Compiling configuration for system {system_name}")
            system_class = ResearchPluginService.get_plugin(system_location)
            configuration_dict[system_name] = system_class.build_pilot_configuration(
                user_configuration.get(system_name, {})
            ).dict()

        return AgentSettings.parse_obj(configuration_dict)

    @classmethod
    async def determine_pilot_name_and_goals(
        cls,
        user_objective: str,
        pilot_settings: AgentSettings,
        logger: logging.Logger,
    ) -> dict:
        logger.debug("Loading OpenAI provider.")
        provider: OpenAIProvider = cls._get_system_instance(
            "openai_provider",
            pilot_settings,
            logger=logger,
        )
        logger.debug("Loading pilot planner.")
        pilot_planner: ResearchPlanner = cls._get_system_instance(
            "planning",
            pilot_settings,
            logger=logger,
            model_providers={"openai": provider},
        )
        logger.debug("determining pilot name and goals.")
        model_response = await pilot_planner.decide_name_and_goals(
            user_objective,
        )

        return model_response.content

    @classmethod
    def provision_pilot(
        cls,
        pilot_settings: AgentSettings,
        logger: logging.Logger,
    ):
        pilot_settings.pilot.configuration.creation_time = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        workspace: ResearchWorkspace = cls._get_system_instance(
            "workspace",
            pilot_settings,
            logger=logger,
        )
        return workspace.setup_workspace(pilot_settings, logger, cls.name())

    @classmethod
    def provision_pilot_goal(
        cls,
        goal: str,
        thread_id: str,
        pilot_settings: AgentSettings,
        logger: logging.Logger,
    ):
        pilot_settings.pilot.configuration.creation_time = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        workspace: ResearchWorkspace = cls._get_system_instance(
            "workspace",
            pilot_settings,
            logger=logger,
        )
        settings_json = workspace.setup_workspace(pilot_settings, logger, cls.name(), save_file=False)
        settings_json = json.loads(settings_json)
        return workspace.save_pilot_settings_goal(goal, thread_id, settings_json)

    @classmethod
    def _get_system_instance(
        cls,
        system_name: str,
        pilot_settings: AgentSettings,
        logger: logging.Logger,
        *args,
        **kwargs,
    ):
        system_locations = pilot_settings.pilot.configuration.systems.dict()

        system_settings = getattr(pilot_settings, system_name)
        system_class = ResearchPluginService.get_plugin(system_locations[system_name])
        system_instance = system_class(
            system_settings,
            *args,
            logger=logger.getChild(system_name),
            **kwargs,
        )
        return system_instance


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
