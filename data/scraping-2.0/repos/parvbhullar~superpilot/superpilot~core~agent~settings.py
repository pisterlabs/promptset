from pydantic import BaseModel
from superpilot.core.ability import AbilityRegistrySettings
from superpilot.core.configuration import SystemConfiguration, SystemSettings
from superpilot.core.configuration.schema import WorkspaceSettings
from superpilot.core.memory.settings import MemorySettings
from superpilot.core.planning.settings import PlannerSettings
from superpilot.core.plugin.base import PluginLocation
from superpilot.core.resource.model_providers.openai import OpenAISettings


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
