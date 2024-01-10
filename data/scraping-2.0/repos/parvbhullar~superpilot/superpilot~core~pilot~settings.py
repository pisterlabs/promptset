from pydantic import BaseModel
from superpilot.core.ability import AbilityRegistrySettings
from superpilot.core.configuration import SystemConfiguration, SystemSettings
from superpilot.core.configuration.schema import WorkspaceSettings
from superpilot.core.memory.settings import MemorySettings
from superpilot.core.planning.settings import PlannerSettings
from superpilot.core.environment.settings import EnvSettings
from superpilot.core.plugin.base import PluginLocation
from superpilot.core.resource.model_providers.openai import OpenAISettings
import enum


class PilotSystems(SystemConfiguration):
    ability_registry: PluginLocation
    environment: PluginLocation
    planning: PluginLocation


class ExecutionAlgo(str, enum.Enum):
    PLAN_AND_EXECUTE = "plan_and_execute"


class PilotConfiguration(SystemConfiguration):
    cycle_count: int
    max_task_cycle_count: int
    creation_time: str
    name: str
    role: str
    goals: list[str]
    # systems: PilotSystems
    # environment: PluginLocation
    execution_algo: ExecutionAlgo


class PilotSystemSettings(SystemSettings):
    configuration: PilotConfiguration


class PilotSettings(BaseModel):
    pilot: PilotSystemSettings
    # environment: EnvSettings
    # ability_registry: AbilityRegistrySettings
    # planning: PlannerSettings

    def update_pilot_name_and_goals(self, pilot_goals: dict) -> None:
        self.pilot.configuration.name = pilot_goals["pilot_name"]
        self.pilot.configuration.role = pilot_goals["pilot_role"]
        self.pilot.configuration.goals = pilot_goals["pilot_goals"]
