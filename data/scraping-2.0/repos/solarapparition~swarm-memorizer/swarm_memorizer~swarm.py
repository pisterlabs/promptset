"""Structure for swarm agents."""

import importlib.util
import sys
import shelve
import asyncio
from itertools import chain
from enum import Enum
from dataclasses import InitVar, dataclass, asdict, field
from functools import cached_property
from pathlib import Path
from uuid import UUID
from typing import (
    Generator,
    Iterable,
    Iterator,
    Literal,
    MutableMapping,
    NewType,
    NamedTuple,
    Any,
    Self,
    Protocol,
    Sequence,
    Set,
    TypeVar,
    Callable,
    Coroutine,
)

from langchain.schema import SystemMessage, BaseMessage
from ruamel.yaml import YAML, YAMLError
from colorama import Fore

from .config import configure_langchain_cache
from .toolkit.models import super_creative_model, precise_model, query_model
from .toolkit.text import ExtractionError, extract_blocks, dedent_and_strip
from .toolkit.yaml_tools import as_yaml_str, default_yaml
from .toolkit.id_generation import (
    utc_timestamp,
    IdGenerator as DefaultIdGenerator,
)

BlueprintId = NewType("BlueprintId", str)
TaskId = NewType("TaskId", str)
EventId = NewType("EventId", str)
DelegatorId = NewType("DelegatorId", str)
RuntimeId = NewType("RuntimeId", str)
TaskHistory = list[TaskId]
IdGenerator = Callable[[], UUID]
IdTypeT = TypeVar("IdTypeT", BlueprintId, TaskId, EventId, DelegatorId)

AGENT_COLOR = Fore.MAGENTA
VERBOSE = True
NONE = "None"


class Concept(Enum):
    """Concepts for swarm agents."""

    MAIN_TASK = "MAIN TASK"
    MAIN_TASK_OWNER = f"{MAIN_TASK} OWNER"
    ORCHESTRATOR = "ORCHESTRATOR"
    ORCHESTRATOR_ACTIONS = f"{ORCHESTRATOR} ACTIONS"
    EXECUTOR = "EXECUTOR"
    RECENT_EVENTS_LOG = "RECENT EVENTS LOG"
    ORCHESTRATOR_INFORMATION_SECTIONS = "ORCHESTRATOR INFORMATION SECTIONS"
    SUBTASK = "SUBTASK"
    SUBTASK_STATUS = f"{SUBTASK} STATUS"
    SUBTASK_EXECUTOR = f"{SUBTASK} {EXECUTOR}"
    FOCUSED_SUBTASK = f"FOCUSED {SUBTASK}"
    FOCUSED_SUBTASK_DISCUSSION_LOG = f"{FOCUSED_SUBTASK} DISCUSSION LOG"
    MAIN_TASK_DESCRIPTION = f"{MAIN_TASK} DESCRIPTION"
    MAIN_TASK_INFORMATION = f"{MAIN_TASK} INFORMATION"
    MAIN_TASK_DEFINITION_OF_DONE = f"{MAIN_TASK} DEFINITION OF DONE"
    TASK_MESSAGES = "TASK MESSAGES"
    LAST_READ_MAIN_TASK_OWNER_MESSAGE = f"LAST READ {MAIN_TASK_OWNER} MESSAGE"


def as_printable(messages: Sequence[BaseMessage]) -> str:
    """Print LangChain messages."""
    return "\n\n---\n\n".join(
        [f"[{message.type.upper()}]:\n\n{message.content}" for message in messages]  # type: ignore
    )


def generate_swarm_id(
    id_type: type[IdTypeT], id_generator: Callable[[], UUID]
) -> IdTypeT:
    """Generate an ID for an agent."""
    return id_type(f"{str(id_generator())}")


class ValidationResult(NamedTuple):
    """Validation of work done by agent."""

    valid: bool
    feedback: str


class Role(Enum):
    """Role of an agent."""

    ORCHESTRATOR = "orchestrator"
    BOT = "bot"


@dataclass
class Reasoning:
    """Reasoning instructions for an agent."""

    default_action_choice: str | None = None
    subtask_action_choice: str | None = None
    subtask_extraction: str | None = None


@dataclass
class OrchestratorBlueprint:
    """A blueprint for an orchestrator."""

    name: str
    rank: int | None
    task_history: TaskHistory
    reasoning: Reasoning
    knowledge: str
    recent_events_size: int
    auto_wait: bool
    id: BlueprintId
    role: Role = field(init=False)

    def __post_init__(self) -> None:
        self.role = Role.ORCHESTRATOR


@dataclass
class BotBlueprint:
    """A blueprint for a bot."""

    name: str
    task_history: TaskHistory
    id: BlueprintId
    kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def role(self) -> Role:
        """Role of the agent."""
        return Role.BOT

    @property
    def rank(self) -> int:
        """Rank of the bot, which is always 0."""
        return 0

    def serialize(self) -> dict[Any, Any]:
        """Serialize the blueprint to a JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def deserialize(cls, data: dict[Any, Any]) -> Self:
        """Deserialize the blueprint from a JSON-compatible dictionary."""
        task_history = [TaskId(task_id) for task_id in data["task_history"]]
        return cls(
            name=data["name"],
            task_history=task_history,
            id=BlueprintId(data["id"]),
            kwargs=data.get("kwargs", {}),
        )


Blueprint = BotBlueprint | OrchestratorBlueprint


class TaskWorkStatus(Enum):
    """Status of the work for a task."""

    IDENTIFIED = "IDENTIFIED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    BLOCKED = "BLOCKED"
    IN_VALIDATION = "IN_VALIDATION"


class TaskEventStatus(Enum):
    """Status of the events for a task."""

    NONE = "None"
    AWAITING_EXECUTOR = "you are awaiting response from the subtask executor"
    AWAITING_OWNER = "the subtask executor is awaiting a response from you"


def replace_agent_id(
    text_to_replace: str, replace_with: str, agent_id: RuntimeId
) -> str:
    """Replace agent id with some other string."""
    return (
        text_to_replace.replace(f"agent {agent_id}", replace_with)
        .replace(f"Agent {agent_id}", replace_with.title())
        .replace(agent_id, replace_with)
    )


def replace_task_id(text_to_replace: str, task_id: TaskId, replacement: str) -> str:
    """Replace task id with some other string."""
    return text_to_replace.replace(task_id, f"`{replacement}`")


@dataclass(frozen=True)
class Message:
    """Data for a message."""

    sender: RuntimeId
    recipient: RuntimeId
    content: str

    def __str__(self) -> str:
        return f"{self.sender} (to {self.recipient}): {self.content}"


@dataclass(frozen=True)
class SubtaskIdentification:
    """Data for identifying a new subtask."""

    owner_id: RuntimeId
    subtask: str
    subtask_id: TaskId
    validation_result: ValidationResult

    def __str__(self) -> str:
        if self.validation_result.valid:
            return f"{self.owner_id}: Successfully identified subtask: `{self.subtask}`; assigned subtask id: `{self.subtask_id}`."
        return f'{self.owner_id}: Attempted to identify subtask `{self.subtask}`, but the validator did not approve the subtask, with the following feedback: "{self.validation_result.feedback}"'


@dataclass(frozen=True)
class TaskStatusChange:
    """Data for changing the status of a subtask."""

    changing_agent: RuntimeId
    task_id: TaskId
    old_status: TaskWorkStatus
    new_status: TaskWorkStatus
    reason: str

    def __str__(self) -> str:
        return f"{self.changing_agent}: I've updated the status of task {self.task_id} from {self.old_status.value} to {self.new_status.value}. Reason: {self.reason.rstrip('.')}."


@dataclass(frozen=True)
class SubtaskFocus:
    """Data for changing the focus of a main task owner."""

    owner_id: RuntimeId
    subtask_id: TaskId

    def __str__(self) -> str:
        return f"{self.owner_id}: I've changed focus to subtask {self.subtask_id}."


@dataclass(frozen=True)
class TaskDescriptionUpdate:
    """Data for updating the description of a task."""

    changing_agent: RuntimeId
    task_id: TaskId
    old_description: str
    new_description: str
    reason: str

    def __str__(self) -> str:
        return f"{self.changing_agent}: I've updated the description of task {self.task_id}. Reason: {self.reason.rstrip('.')}."


@dataclass(frozen=True)
class Thought:
    """Data for a thought."""

    agent_id: RuntimeId
    content: str

    def __str__(self) -> str:
        return f"{self.agent_id} (Thought): {self.content}"


EventData = (
    Message
    | SubtaskIdentification
    | TaskStatusChange
    | SubtaskFocus
    | TaskDescriptionUpdate
    | Thought
)


@dataclass
class Event:
    """An event in the event log."""

    data: EventData
    generating_task_id: TaskId
    """Id of the task that generated the event."""
    id: EventId
    timestamp: str = field(default_factory=utc_timestamp)

    # @cached_property
    # def id(self) -> EventId:
    #     """Id of the event."""
    #     return generate_swarm_id(EventId, self.id_generator)

    def __str__(self) -> str:
        # return f"[{self.timestamp}] {self.data}"
        return f"{self.data}"

    def to_str_with_pov(
        self,
        pov_id: RuntimeId,
        other_id: RuntimeId,
        other_name: str,
        task_id_replacement: dict[TaskId, str] | None = None,
        subtask_executor_replacement: dict[RuntimeId, str] | None = None,
    ) -> str:
        """String representation of the event with a point of view from a certain executor."""
        event_printout = replace_agent_id(str(self), "You", pov_id)
        event_printout = replace_agent_id(event_printout, other_name, other_id)
        if not task_id_replacement:  # or not isinstance(self.data, TaskStatusChange):
            return event_printout
        for task_id, replacement in task_id_replacement.items():
            event_printout = replace_task_id(event_printout, task_id, replacement)
        if not subtask_executor_replacement:
            return event_printout
        for executor_id, replacement in subtask_executor_replacement.items():
            event_printout = replace_agent_id(event_printout, replacement, executor_id)
        return event_printout

    def serialize(self) -> dict[str, Any]:
        """Serialize the event."""
        return asdict(self)

    def __repr__(self) -> str:
        """String representation of the event."""
        return str(self.serialize())


class WorkValidator(Protocol):
    """A validator of a task."""

    def validate(self, prompt: str) -> ValidationResult:
        """Validate the work done by an executor for a task."""
        raise NotImplementedError


@dataclass
class Human:
    """A human part of the hivemind. Can be slotted into various specialized roles for tasks that the agent can't yet handle autonomously."""

    name: str = "Human"
    reply_cache: MutableMapping[str, str] | None = None
    thread: list[str] = field(default_factory=list)

    @property
    def id(self) -> RuntimeId:
        """Runtime id of the human."""
        return RuntimeId(self.name)

    def respond_manually(self) -> str:
        """Get manual response from the human."""
        return input("Enter your response: ").strip()

    def respond_using_cache(self, reply_cache: MutableMapping[str, str]) -> str:
        """Get cached reply based on thread."""
        if reply := reply_cache.get(str(self.thread)):
            print(f"Cached reply found: {reply}")
            return reply
        if reply := self.respond_manually():
            reply_cache.update({str(self.thread): reply})
        return reply

    def advise(self, prompt: str) -> str:
        """Get input from the human."""
        print(prompt)
        self.thread.append(prompt)
        self.thread.append(
            reply := (
                self.respond_using_cache(self.reply_cache)
                if self.reply_cache is not None
                else self.respond_manually()
            )
        )
        return reply

    def validate(self, prompt: str) -> ValidationResult:
        """Validate the work done by an executor."""
        prompt += "\n\nPlease validate the work as described above (y/n): "
        while True:
            validation_input: str = self.advise(prompt).strip().lower()
            if validation_input in {"y", "n"}:
                validated: bool = validation_input == "y"
                break
            print("Invalid input. Please enter 'y' or 'n'.")
        feedback: str = "" if validated else self.advise("Provide feedback: ")
        return ValidationResult(validated, feedback)


@dataclass
class TaskList:
    """A list of tasks and their managment functionality."""

    items: list["Task"] = field(default_factory=list)

    def __str__(self) -> str:
        """String representation of the task list."""
        # if we're printing out the whole task list, assume these are subtasks
        return "\n".join([task.as_subtask_printout for task in self.items]) or NONE

    def __iter__(self) -> Iterator["Task"]:
        """Iterate over the task list."""
        return iter(self.items)

    def __bool__(self) -> bool:
        """Whether the task list is empty."""
        return bool(self.items)

    def filter_by_status(self, status: TaskWorkStatus) -> "TaskList":
        """Filter the task list by status."""
        return TaskList(
            items=[task for task in self.items if task.work_status == status]
        )


@dataclass
class EventLog:
    """A log of events within a task."""

    events: list[Event] = field(default_factory=list)

    @property
    def last_event(self) -> Event | None:
        """Last event in the event log."""
        return self.events[-1] if self.events else None

    @property
    def messages(self) -> "EventLog":
        """Messages in the event log."""
        return EventLog(
            events=[event for event in self.events if isinstance(event.data, Message)]
        )

    def to_str_with_pov(
        self,
        pov_id: RuntimeId,
        other_id: RuntimeId,
        other_name: str,
        task_id_replacement: dict[TaskId, str] | None = None,
        subtask_executor_replacement: dict[RuntimeId, str] | None = None,
    ) -> str:
        """String representation of the event log with a point of view from a certain executor."""
        return (
            "\n".join(
                [
                    event.to_str_with_pov(
                        pov_id,
                        other_id,
                        other_name,
                        task_id_replacement,
                        subtask_executor_replacement,
                    )
                    for event in self.events
                ]
            )
            if self.events
            else NONE
        )

    def recent(self, num_recent: int) -> "EventLog":
        """Recent events."""
        return EventLog(events=self.events[-num_recent:])

    def add(self, *events: Event) -> None:
        """Add events to the event log."""
        self.events.extend(events)

    def __str__(self) -> str:
        """String representation of the event log."""
        return "\n".join([str(event) for event in self.events]) if self.events else NONE

    def __bool__(self) -> bool:
        """Whether the event log is empty."""
        return bool(self.events)

    def __iter__(self) -> Iterator[Event]:
        """Iterate over the event log."""
        return iter(self.events)


@dataclass
class TaskDescription:
    """Description of a task."""

    information: str
    definition_of_done: str | None = None

    def __str__(self) -> str:
        """String representation of the task description."""
        template = """
        Information:
        {information}

        Definition of Done:
        {definition_of_done}
        """
        return dedent_and_strip(template).format(
            information=self.information, definition_of_done=self.definition_of_done
        )


@dataclass
class ExecutorReport:
    """Report from an executor."""

    reply: str
    events: list[Event] = field(default_factory=list)


class Executor(Protocol):
    """An agent responsible for executing a task."""

    @property
    def blueprint(self) -> Blueprint:
        """Blueprint of the executor."""
        raise NotImplementedError

    # blueprint: Blueprint

    @property
    def id(self) -> RuntimeId:
        """Runtime id of the executor."""
        raise NotImplementedError

    @property
    def rank(self) -> int | None:
        """Rank of the executor."""
        raise NotImplementedError

    def accepts(self, task: "Task") -> bool:
        """Decides whether the executor accepts a task."""

        breakpoint()
        raise NotImplementedError

    async def execute(self, message: str | None = None) -> ExecutorReport:
        """Execute the task. Adds a message to the task's event log if provided, and adds own message to the event log at the end of execution."""
        raise NotImplementedError


@dataclass
class Task:
    """Holds information about a task."""

    description: TaskDescription
    owner_id: RuntimeId
    rank_limit: int | None
    validator: WorkValidator
    id_generator: IdGenerator
    name: str | None = None
    executor: Executor | None = None
    notes: dict[str, str] = field(default_factory=dict)
    work_status: TaskWorkStatus = TaskWorkStatus.IDENTIFIED
    # event_status: TaskEventStatus = TaskEventStatus.NONE

    @cached_property
    def id(self) -> TaskId:
        """Id of the task."""
        return generate_swarm_id(TaskId, self.id_generator)

    @property
    def definition_of_done(self) -> str | None:
        """Definition of done for the task."""
        return self.description.definition_of_done

    @property
    def information(self) -> str:
        """Information on the task."""
        return self.description.information

    @cached_property
    def event_log(self) -> EventLog:
        """Event log for the task."""
        return EventLog()

    @cached_property
    def subtasks(self) -> TaskList:
        """Subtasks of the task."""
        return TaskList()

    @property
    def messages(self) -> EventLog:
        """Messages for the task."""
        return self.event_log.messages

    @property
    def as_main_task_printout(self) -> str:
        """String representation of the task as it would appear as a main task."""
        return str(self.description)

    def __str__(self) -> str:
        """String representation of task status."""
        return self.as_main_task_printout

    @property
    def executor_id(self) -> RuntimeId | None:
        """Id of the task's executor."""
        return self.executor.id if self.executor else None

    @property
    def executor_blueprint_id(self) -> BlueprintId | None:
        """Id of the task's executor."""
        return self.executor.blueprint.id if self.executor else None

    @property
    def as_subtask_printout(self) -> str:
        """String representation of task as it would appear as a subtask."""
        if self.work_status in {TaskWorkStatus.COMPLETED, TaskWorkStatus.CANCELLED}:
            template = """
            - Id: {id}
              Name: {name}
            """
            return dedent_and_strip(template).format(
                id=self.id,
                name=self.name,
            )
        template = """
        - Id: {id}
          Name: {name}
          Work Status: {work_status}
        """
        return dedent_and_strip(template).format(
            id=self.id,
            name=self.name,
            work_status=self.work_status.value,
            # event_status=self.event_status.value,
        )

    def reformat_event_log(
        self,
        event_log: EventLog,
        pov: Literal[Concept.EXECUTOR, Concept.MAIN_TASK_OWNER],
    ) -> str:
        """Format an event log."""
        assert self.executor_id is not None

        # use case for executor pov is for printing recent events log
        if pov == Concept.EXECUTOR:
            task_id_replacement = {
                self.id: Concept.MAIN_TASK.value,
            }
            subtask_executor_replacement = {
                subtask.executor_id: f"{Concept.EXECUTOR.value} for subtask {subtask.id}"
                for subtask in self.subtasks
                if subtask.executor_id is not None
            }
            return event_log.to_str_with_pov(
                pov_id=self.executor_id,
                other_id=self.owner_id,
                other_name=Concept.MAIN_TASK_OWNER.value,
                task_id_replacement=task_id_replacement,
                subtask_executor_replacement=subtask_executor_replacement,
            )

        # use case for main task owner pov is for printing subtask discussion log when focused on a subtask
        assert (
            self.name is not None
        ), "If POV in a task discussion is from the main task owner, the task must have a name."
        task_id_replacement = None
        return event_log.to_str_with_pov(
            pov_id=self.owner_id,
            other_id=self.executor_id,
            other_name=Concept.EXECUTOR.value,
        )

    def discussion(
        self, pov: Literal[Concept.EXECUTOR, Concept.MAIN_TASK_OWNER]
    ) -> str:
        """Discussion of a task in the event log."""
        return self.reformat_event_log(self.event_log.messages, pov)


@dataclass
class CoreState:
    """Core runtime state of an orchestrator."""

    id: RuntimeId
    knowledge: str
    main_task: Task
    subtasks: TaskList
    template: str

    def __str__(self) -> str:
        """String representation of the core state."""
        completed_subtasks = str(
            self.subtasks.filter_by_status(TaskWorkStatus.COMPLETED)
        )
        cancelled_subtasks = str(
            self.subtasks.filter_by_status(TaskWorkStatus.CANCELLED)
        )
        in_validation_subtasks = str(
            self.subtasks.filter_by_status(TaskWorkStatus.IN_VALIDATION)
        )
        delegated_subtasks = str(
            self.subtasks.filter_by_status(TaskWorkStatus.IN_PROGRESS)
        )
        blocked_subtasks = str(self.subtasks.filter_by_status(TaskWorkStatus.BLOCKED))
        return dedent_and_strip(self.template).format(
            knowledge=self.knowledge,
            task_specification=str(self.main_task),
            completed_subtasks=completed_subtasks,
            cancelled_subtasks=cancelled_subtasks,
            in_validation_subtasks=in_validation_subtasks,
            delegated_subtasks=delegated_subtasks,
            # identified_subtasks=identified_subtasks,
            blocked_subtasks=blocked_subtasks,
        )


class ActionName(Enum):
    """Names of actions available to the orchestrator."""

    IDENTIFY_NEW_SUBTASK = "IDENTIFY_NEW_SUBTASK"
    START_DISCUSSION_FOR_SUBTASK = "START_DISCUSSION_FOR_SUBTASK"
    MESSAGE_TASK_OWNER = "ASK_MAIN_TASK_OWNER"
    REPORT_MAIN_TASK_COMPLETE = "REPORT_MAIN_TASK_COMPLETE"
    WAIT = "WAIT"
    MESSAGE_SUBTASK_EXECUTOR = "MESSAGE_SUBTASK_EXECUTOR"
    PAUSE_SUBTASK_DISCUSSION = "PAUSE_SUBTASK_DISCUSSION"
    CANCEL_SUBTASK = "CANCEL_SUBTASK"


ORCHESTRATOR_CONCEPTS = f"""
- {Concept.ORCHESTRATOR.value}: the agent that is responsible for managing the execution of a main task and managing the statuses of its subtasks, while communicating with the task's owner to gather required information for the task. The orchestrator must communicate with both the task owner and subtask executors to complete the main task as efficiently as possible.
- {Concept.MAIN_TASK.value}: the main task that the orchestrator is responsible for managing, which it does by identifying subtasks and providing support for specialized executor agents for the subtasks.
- {Concept.SUBTASK.value}: a task that must be executed in order to complete the main task. The orchestrator does NOT execute subtasks itself; instead, it facilitates the resolution of subtasks by making high-level decisions regarding each subtask in the context of the overall task and providing support for the subtask executors.
- SUBTASK STATUS: the status of subtasks that have already been identified. The status of a subtask can be one of the following:
  - {TaskWorkStatus.BLOCKED.value}: the subtask is blocked by some issue, and execution cannot continue until the issue is resolved, typically by discussing the blocker and/or identifying a new subtask to resolve the blocker.
  - {TaskWorkStatus.IN_PROGRESS.value}: the subtask is currently being executed by a subtask executor.
  - {TaskWorkStatus.IN_VALIDATION.value}: the subtask has been reported as completed by its executor, but is still being validated by a validator. Validation happens automatically and does not require or action from the orchestrator.
  - {TaskWorkStatus.COMPLETED.value}: the subtask has been validated as complete by a validator. Completed subtasks provide a record of overall successful progress for the main task.
  - {TaskWorkStatus.CANCELLED.value}: the subtask has been cancelled for various reason and will not be done.
- SUBTASK EXECUTOR: an agent that is responsible for executing a subtask. Subtask executors specialize in executing certain types of tasks; whenever a subtask is identified, an executor is automatically assigned to it without any action required from the orchestrator.
- {Concept.MAIN_TASK_OWNER.value}: the one who requested the main task to be done. The orchestrator must communicate with the task owner to gather background information required to complete the main task.
""".strip()


def query_and_extract_reasoning(
    messages: Sequence[SystemMessage], preamble: str, printout: bool
) -> str:
    """Query the model and extract the reasoning process."""
    if printout:
        result = query_model(
            model=super_creative_model,
            messages=messages,
            preamble=preamble,
            color=AGENT_COLOR,
            printout=printout,
        )
    else:
        result = query_model(
            model=super_creative_model,
            messages=messages,
            printout=printout,
        )
    if not (extracted_result := extract_blocks(result, "start_of_reasoning_process")):
        raise ExtractionError("Could not extract reasoning process from the result.")
    return extracted_result[0]


class ActionModeName(Enum):
    """States of an action."""

    DEFAULT = "DEFAULT"
    SUBTASK_DISCUSSION = "SUBTASK DISCUSSION"


ORCHESTRATOR_INSTRUCTOR_MISSION = """
You are the instructor for an AI task orchestration agent. Your purpose is to provide step-by-step guidance for the agent to think through what it must do next.""".strip()

MODULAR_SUBTASK_IDENTIFICATION = """
"Modular Subtask Identification" (MSI) is a philosophy for identifying a required subtask from a main task that emphasizes two principles:
- orthogonality: the identified subtask is as independent from the rest of the uncompleted main task as possible. This allows it to be executed in isolation without requiring any other subtasks to be completed first.
- small input/output footprint: the identified subtask has a small input and output footprint, meaning that it requires little information to be provided to it, and provides compact output. This reduces the amount of context needed to understand the subtask and its results.
""".strip()


@dataclass(frozen=True)
class ActionDecision:
    """Decision for an action."""

    action_choice: str
    justifications: str
    additional_thoughts: str | None = None

    @cached_property
    def action_name(self) -> str:
        """Name of the action chosen."""
        return (
            self.action_choice.split(":")[0]
            if ":" in self.action_choice
            else self.action_choice
        )

    @cached_property
    def action_args(self) -> dict[str, str]:
        """Arguments of the action chosen."""
        action_args: dict[str, str] = {}
        if self.action_name in [
            ActionName.MESSAGE_TASK_OWNER.value,
            ActionName.MESSAGE_SUBTASK_EXECUTOR.value,
        ]:
            action_args["message"] = (
                self.action_choice.replace(f"{self.action_name}:", "")
                .strip()
                .strip('"')
            )
            # self.action_choice.split(":")[1].strip().strip('"')
        return action_args

    @classmethod
    def from_yaml_str(cls, yaml_str: str) -> Self:
        """Create an action decision from a YAML string."""
        data = default_yaml.load(yaml_str)
        if (
            additional_thoughts := data.get("additional_thoughts")
        ) and "NONE" in additional_thoughts:
            data["additional_thoughts"] = None
        return cls(**data)

    def validate_action(self, valid_actions: Iterable[str]) -> None:
        """Validate that the action is allowed."""
        for allowed_action in valid_actions:
            if self.action_choice.startswith(allowed_action):
                return
        raise ValueError(
            "Action choice validation failed.\n"
            f"{valid_actions=}\n"
            f"{self.action_choice=}\n"
        )


PauseExecution = NewType("PauseExecution", bool)


@dataclass
class ActionResult:
    """Result of an action."""

    pause_execution: PauseExecution
    new_events: list[Event]
    new_work_status: TaskWorkStatus | None = None
    # new_event_status: TaskEventStatus | None = None


class OrchestratorReasoningNotes(Enum):
    """Notes for action reasoning."""

    OVERVIEW = "Provide a step-by-step, robust reasoning process for the orchestrator to sequentially think through the information it has access to so that it has the appropriate mental context for deciding what to do next. These steps provide the internal thinking that an intelligent agent must go through so that they have all the relevant information on top of mind. Some things to note:"
    ACTION_RESTRICTIONS = f"The final action that the orchestrator decides on MUST be one of the {Concept.ORCHESTRATOR_ACTIONS.value} described above. The orchestrator cannot perform any other actions."
    FOCUSED_SUBTASK_RESTRICTIONS = f"The orchestrator cannot directly change the {Concept.FOCUSED_SUBTASK.value}. To focus on a different subtask, it must first use the {ActionName.PAUSE_SUBTASK_DISCUSSION.value} action first. Overall, the orchestrator should be focused on helping the EXECUTOR of the {Concept.FOCUSED_SUBTASK.value}, and will need strong reason to change its focus."
    INFORMATION_RESTRICTIONS = f"Assume that the orchestrator has access to what's described in {Concept.ORCHESTRATOR_INFORMATION_SECTIONS.value} above, but no other information, except for general world knowledge that is available to a standard LLM like GPT-3."
    TERM_REFERENCES = """The orchestrator requires precise references to information it's been given, and it may need a reminder to check for specific parts; it's best to be explicit and use the _exact_ capitalized terminology to refer to concepts or information sections (e.g. "MAIN TASK" or "KNOWLEDGE section"); however, only use capitalization to refer to specific terms—don't use capitalization as emphasis, as that could be confusing to the orchestrator."""
    SUBTASK_STATUS_INFO = f"Typically, subtasks that are {TaskWorkStatus.COMPLETED.value}, {TaskWorkStatus.CANCELLED.value}, {TaskWorkStatus.IN_PROGRESS.value}, or {TaskWorkStatus.IN_VALIDATION.value} do not need immediate attention unless the orchestrator discovers information that changes the status of the subtask. Subtasks that are {TaskWorkStatus.BLOCKED.value} will need action from the orchestrator to start or resume execution respectively."
    STEPS_RESTRICTIONS = "The reasoning process should be written in second person and be around 5-7 steps, though you can add substeps within a step (a, b, c, etc.) if it is complex."
    PROCEDURAL_SCRIPTING = "The reasoning steps can refer to the results of previous steps, and it may be effective to build up the orchestrator's mental context step by step, starting from basic information available, similar to writing a procedural script for a program but in natural language instead of code."


REASONING_OUTPUT_INSTRUCTIONS = """
Provide the reasoning process in the following format:
```start_of_reasoning_process
1. {reasoning step 1}
2. {reasoning step 2}
3. [... etc.]
```end_of_reasoning_process
You may add comments or thoughts before or after the reasoning process, but the reasoning process block itself must only contain the reasoning steps, directed at the orchestrator.
""".strip()


class OrchestratorInformationSection(Enum):
    """Information sections available to orchestrators."""

    KNOWLEDGE = "KNOWLEDGE: background knowledge relating to the orchestrator's area of specialization. The information may or may not be relevant to the specific main task, but is provided as support for the orchestrator's decisionmaking."
    MAIN_TASK_DESCRIPTION = f"MAIN TASK DESCRIPTION: a description of information about the main task that the orchestrator has learned so far from the {Concept.MAIN_TASK_OWNER.value}. This may NOT be a complete description of the main task, so the orchestrator must always take into account if there is enough information for performing its actions. Additional information may also be in the {Concept.RECENT_EVENTS_LOG.value}, as messages from the main task owner."
    SUBTASKS = "SUBTASKS: a list of all subtasks that have been identified by the orchestrator so far; for each one, there is a high-level description of what must be done, as well as the subtask's status. This is not an exhaustive list of all required subtasks for the main task; there may be additional subtasks that are required. This list is automatically maintained and updated by a background process."
    RECENT_EVENTS_LOG = f"{Concept.RECENT_EVENTS_LOG.value}: a log of recent events that have occurred during the execution of the task. This can include status updates for subtasks, messages from the main task owner, and the orchestrator's own previous thoughts/decisions."
    FOCUSED_SUBTASK = f"{Concept.FOCUSED_SUBTASK.value}: the subtask that the orchestrator is currently focused on. This is the subtask that the orchestrator is currently thinking about and making decisions for. The orchestrator can only focus on one subtask at a time, and cannot perform actions on subtasks that it is not currently focused on."
    FOCUSED_SUBTASK_FULL_DISCUSSION_LOG = f"{Concept.FOCUSED_SUBTASK_DISCUSSION_LOG.value}: a log of the full discussion for the focused subtask between the orchestrator and the subtask executor."


@dataclass(frozen=True)
class SubtaskIdentifcationResult:
    """Result of subtask identification."""

    identified_subtask: str
    additional_thoughts: str | None = None

    @classmethod
    def from_yaml_str(cls, yaml_str: str) -> Self:
        """Create a subtask identification result from a YAML string."""
        data = default_yaml.load(yaml_str)
        if (
            additional_thoughts := data.get("additional_thoughts")
        ) and "NONE" in additional_thoughts:
            data["additional_thoughts"] = None
        return cls(**data)


@dataclass
class ReasoningGenerator:
    """Generates reasoning for an orchestrator."""

    _orchestrator: "Orchestrator"
    """Orchestrator for which to generate reasoning. Must not be modified."""

    @property
    def base_info(self) -> str:
        """Base information for the orchestrator."""
        return self._orchestrator.base_info

    @property
    def default_mode_actions(self) -> str:
        """Actions available to the orchestrator in the default state."""
        return self._orchestrator.default_mode_actions

    @property
    def role(self) -> Role:
        """Role of the orchestrator."""
        return self._orchestrator.role

    @property
    def subtask_mode_actions(self) -> str:
        """Actions available to the orchestrator in the subtask discussion state."""
        return self._orchestrator.subtask_mode_actions

    def generate_default_action_reasoning(self) -> str:
        """Generate reasoning for choosing an action in the default state."""
        context = """
        ## MISSION:
        {mission}

        {base_info}

        ## {ORCHESTRATOR_ACTIONS}:
        In its default state, the orchestrator can perform the following actions:
        {actions}
        """
        request = f"""
        ## REQUEST FOR YOU:
        {OrchestratorReasoningNotes.OVERVIEW.value}
        - {OrchestratorReasoningNotes.ACTION_RESTRICTIONS.value}
        - {OrchestratorReasoningNotes.INFORMATION_RESTRICTIONS.value}
        - {OrchestratorReasoningNotes.TERM_REFERENCES.value}
        - {OrchestratorReasoningNotes.SUBTASK_STATUS_INFO.value}
        - {OrchestratorReasoningNotes.STEPS_RESTRICTIONS.value}
        - {OrchestratorReasoningNotes.PROCEDURAL_SCRIPTING.value}

        {{output_instructions}}
        """
        messages = [
            SystemMessage(
                content=dedent_and_strip(context).format(
                    mission=ORCHESTRATOR_INSTRUCTOR_MISSION,
                    base_info=self.base_info,
                    ORCHESTRATOR_ACTIONS=Concept.ORCHESTRATOR_ACTIONS.value,
                    actions=self.default_mode_actions,
                )
            ),
            SystemMessage(
                content=dedent_and_strip(request).format(
                    output_instructions=REASONING_OUTPUT_INSTRUCTIONS,
                )
            ),
        ]
        return query_and_extract_reasoning(
            messages,
            preamble=f"Generating reasoning for {self.role.value} in {ActionModeName.DEFAULT.value} state...\n{as_printable(messages)}",
            printout=VERBOSE,
        )

    def generate_subtask_action_reasoning(self) -> str:
        """Generate reasoning for choosing an action in the subtask discussion state."""
        context = """
        ## MISSION:
        {mission}

        {base_info}

        ## {ORCHESTRATOR_ACTIONS}:
        The orchestrator is currently in a mode where it is discussing its FOCUSED SUBTASK with the SUBTASK EXECUTOR. Currently, the orchestrator can perform the following actions:
        {actions}
        """

        request = f"""
        ## REQUEST FOR YOU:
        {OrchestratorReasoningNotes.OVERVIEW.value}
        - {OrchestratorReasoningNotes.ACTION_RESTRICTIONS.value}
        - {OrchestratorReasoningNotes.INFORMATION_RESTRICTIONS.value}
        - {OrchestratorReasoningNotes.TERM_REFERENCES.value}
        - {OrchestratorReasoningNotes.SUBTASK_STATUS_INFO.value}
        - {OrchestratorReasoningNotes.FOCUSED_SUBTASK_RESTRICTIONS.value}
        - {OrchestratorReasoningNotes.STEPS_RESTRICTIONS.value}
        - {OrchestratorReasoningNotes.PROCEDURAL_SCRIPTING.value}
        
        {{output_instructions}}
        """
        messages = [
            SystemMessage(
                content=dedent_and_strip(context).format(
                    mission=ORCHESTRATOR_INSTRUCTOR_MISSION,
                    base_info=self.base_info,
                    ORCHESTRATOR_ACTIONS=Concept.ORCHESTRATOR_ACTIONS.value,
                    actions=self.subtask_mode_actions,
                )
            ),
            SystemMessage(
                content=dedent_and_strip(request).format(
                    output_instructions=REASONING_OUTPUT_INSTRUCTIONS,
                )
            ),
        ]
        return query_and_extract_reasoning(
            messages,
            preamble=f"Generating reasoning for {self.role.value} in {ActionModeName.DEFAULT.value} state...\n{as_printable(messages)}",
            printout=VERBOSE,
        )

    def generate_subtask_identification_reasoning(self) -> str:
        """Generate reasoning for identifying a new subtask."""
        context = """
        ## MISSION:
        You are the instructor for an AI task orchestration agent. Your purpose is to provide step-by-step guidance for the agent to think through how to identify the next subtask from the main task description.

        {base_info}

        ## MODULAR SUBTASK INDENTIFICATION PHILOSOPHY:
        {msi}

        """
        request = f"""
        ## REQUEST FOR YOU:
        Provide a step-by-step, robust reasoning process for the orchestrator to a) understand what MSI is and follow its principles, and b) sequentially process the information in the information sections it has access to so that it can identify a new subtask that is not yet identified. These steps provide the internal thinking that an intelligent agent must go through so that they have all the relevant information on top of mind before they perform subtask identification. Some things to note:
        - {OrchestratorReasoningNotes.INFORMATION_RESTRICTIONS.value}
        - {OrchestratorReasoningNotes.TERM_REFERENCES.value}
        - In its current state, the orchestrator is not able to perform any other actions besides subtask identification and the reasoning preceeding it.
        - {OrchestratorReasoningNotes.STEPS_RESTRICTIONS.value}
        - {OrchestratorReasoningNotes.PROCEDURAL_SCRIPTING.value}
        - The orchestrator should only perform the subtask identification on the _last_ step, after it has considered _all_ the information it needs. No other actions need to be performed after subtask identification.
        {{output_instructions}}
        """
        messages = [
            SystemMessage(
                content=dedent_and_strip(context).format(
                    base_info=self.base_info,
                    msi=MODULAR_SUBTASK_IDENTIFICATION,
                )
            ),
            SystemMessage(
                content=dedent_and_strip(request).format(
                    output_instructions=REASONING_OUTPUT_INSTRUCTIONS
                )
            ),
        ]
        return query_and_extract_reasoning(
            messages,
            preamble=f"Generating subtask extraction reasoning...\n{as_printable(messages)}",
            printout=VERBOSE,
        )

    @staticmethod
    def generate_main_task_update_reasoning(printout: bool = True) -> str:
        """Generate reasoning for updating the main task. Currently unused."""
        context = f"""
        ## MISSION:
        You are the instructor for an AI task orchestration agent. Your purpose is to provide step-by-step guidance for the agent to think through how to update the main task description based on new information in an event log.

        ## CONCEPTS:
        These are the concepts you should be familiar with:
        - {Concept.ORCHESTRATOR.value}: the agent that is responsible for managing the execution of a main task while communicating with the {Concept.MAIN_TASK_OWNER.value} to gather required information for the task.
        - {Concept.MAIN_TASK_OWNER.value}: the agent that owns the main task and is responsible for providing information to the orchestrator to help it execute the main task.
        - {Concept.MAIN_TASK.value}: the task that the orchestrator is responsible for executing.

        ## {Concept.ORCHESTRATOR_INFORMATION_SECTIONS.value}:
        The orchestrator has access to the following information:
        - {Concept.MAIN_TASK_DEFINITION_OF_DONE.value}: a description of the criteria that must be met for the {Concept.MAIN_TASK.value} to be considered complete.
        - {Concept.MAIN_TASK_INFORMATION.value}: information on what the {Concept.MAIN_TASK.value} is about, including the goal of the task and any relevant background information. This section provides details that may be too granular for the {Concept.MAIN_TASK_DEFINITION_OF_DONE.value} section.
        - {Concept.TASK_MESSAGES.value}: a transcript of the messages between the orchestrator and the {Concept.MAIN_TASK_OWNER.value}.
        - {Concept.LAST_READ_MAIN_TASK_OWNER_MESSAGE.value}: the last message in the {Concept.TASK_MESSAGES.value} section sent by the {Concept.MAIN_TASK_OWNER.value} that has been read by the orchestrator. All messages after this message have not been read by the orchestrator yet.
        """

        task = f"""
        ## REQUEST FOR YOU:
        Provide a step-by-step, robust reasoning process for the orchestrator to sequentially think through the information it has access to so that it has the appropriate mental context for updating the {Concept.MAIN_TASK_INFORMATION.value} and {Concept.MAIN_TASK_DEFINITION_OF_DONE.value} sections to reflect the new information in the {Concept.TASK_MESSAGES.value} that comes after {Concept.LAST_READ_MAIN_TASK_OWNER_MESSAGE.value}. These steps provide the internal thinking that an intelligent agent must go through so that they have all the relevant information on top of mind. Some things to note:
        - This reasoning process does not make the actual updates to the {Concept.MAIN_TASK_INFORMATION.value} and {Concept.MAIN_TASK_DEFINITION_OF_DONE.value} sections; it only figures out what updates are needed.
        - Both the {Concept.MAIN_TASK_INFORMATION} and {Concept.MAIN_TASK_DEFINITION_OF_DONE} sections may be outdated, hence the need to update them with the latest messages from the {Concept.MAIN_TASK_OWNER.value}.
        - {OrchestratorReasoningNotes.INFORMATION_RESTRICTIONS.value}
        - The orchestrator requires precise references to information it's been given, and it may need a reminder to check for specific parts; it's best to be explicit and use the _exact_ capitalized terminology to refer to concepts or information sections (e.g. "{Concept.MAIN_TASK.value}" or "{Concept.TASK_MESSAGES.value} section"); however, don't use capitalization as emphasis for any other terms.
        - {OrchestratorReasoningNotes.STEPS_RESTRICTIONS.value}
        - {OrchestratorReasoningNotes.PROCEDURAL_SCRIPTING.value}

        {{output_instructions}}
        """
        messages = [
            SystemMessage(content=dedent_and_strip(context)),
            SystemMessage(
                content=dedent_and_strip(task).format(
                    output_instructions=REASONING_OUTPUT_INSTRUCTIONS
                )
            ),
        ]
        return query_and_extract_reasoning(
            messages,
            preamble=f"Generating reasoning for updating the main task...\n{as_printable(messages)}",
            printout=printout,
        )


@dataclass
class OrchestratorState:
    """State of the orchestrator."""

    focused_subtask: Task | None = None
    new_event_count: int = 0


@dataclass(frozen=True)
class Orchestrator:
    """A recursively auto-specializing swarm agent."""

    blueprint: OrchestratorBlueprint
    task: Task
    files_parent_dir: Path
    delegator: "Delegator"
    state: OrchestratorState = field(default_factory=OrchestratorState)

    @classmethod
    @property
    def default_recent_events_size(cls) -> int:
        """Default size of recent events."""
        return 10

    @property
    def focused_subtask(self) -> Task | None:
        """Focused subtask of the orchestrator."""
        return self.state.focused_subtask

    @property
    def new_event_count(self) -> int:
        """Number of new events."""
        return self.state.new_event_count

    @property
    def id(self) -> RuntimeId:
        """Runtime id of the orchestrator."""
        return RuntimeId(f"{self.blueprint.id}_{self.task.id}")

    @property
    def executor_max_rank(self) -> int | None:
        """Maximum rank of the orchestrator's task executors."""
        executors = [subtask.executor for subtask in self.task.subtasks]
        ranks = [
            executor.rank
            for executor in executors
            if executor is not None and executor.rank is not None
        ]
        # if filtered ranks is less than num executors, it means some task have either not been delegated or its executor is unranked
        return None if len(ranks) != len(executors) else max(ranks)

    @property
    def rank_limit(self) -> int | None:
        """Limit of how high the orchestrator can be ranked."""
        return self.task.rank_limit

    @property
    def rank(self) -> int | None:
        """Rank of the orchestrator."""
        # we always go with existing rank if available b/c executor_max_rank varies and could be < existing rank between runs
        if self.blueprint.rank is not None:
            return self.blueprint.rank
        if (
            rank := None
            if self.executor_max_rank is None
            else 1 + self.executor_max_rank
        ) is not None:
            assert (
                rank >= 1
            ), f"Orchestrator rank must be >= 1. For {self.id}, rank={rank}."
            if self.rank_limit is not None:
                rank = min(rank, self.rank_limit)
        return rank

    @property
    def task_history(self) -> TaskHistory:
        """History of tasks completed by the orchestrator."""
        return self.blueprint.task_history

    @property
    def reasoning(self) -> Reasoning:
        """Instructions for the orchestrator for various tasks."""
        return self.blueprint.reasoning

    @property
    def knowledge(self) -> str:
        """Learnings from past tasks."""
        return self.blueprint.knowledge or NONE

    @property
    def role(self) -> Role:
        """Role of the orchestrator."""
        return self.blueprint.role

    @property
    def core_template(self) -> str:
        """Template for the core state."""
        template = f"""
        ## MISSION:
        You are an advanced task orchestrator that specializes in managing the execution of a MAIN TASK and delegating its SUBTASKS to EXECUTORS that can execute those tasks, while communicating with the MAIN TASK OWNER to gather required information for the task. Your overall purpose is to facilitate task execution by communicating with both the MAIN TASK OWNER and SUBTASK EXECUTORS to complete the MAIN TASK as efficiently as possible.

        ## KNOWLEDGE:
        In addition to the general background knowledge of your language model, you have the following, more specialized knowledge that may be relevant to the task at hand:
        ```start_of_knowledge
        {{knowledge}}
        ```end_of_knowledge

        ## MAIN TASK DESCRIPTION:
        Here is information about the main task you are currently working on:
        ```start_of_main_task_description
        {{task_specification}}
        ```end_of_main_task_description
        More recent information may be available in the RECENT EVENTS LOG below. These will be automatically integrated into the main task description when they are no longer recent.

        ## SUBTASKS:
        - SUBTASKS are tasks that must be executed in order to complete the MAIN TASK.
        - You do NOT execute subtasks yourself, but instead delegate them to SUBTASK EXECUTORS.
        - Typically, tasks that are COMPLETED, CANCELLED, IN_PROGRESS, or IN_VALIDATION do not need attention unless you discover information that changes the status of the subtask.
        - In contrast, tasks that are NEW or BLOCKED will need action from you to start/continue execution.
        - This is not an exhaustive list of all required subtasks for the main task; you may discover additional subtasks that must be done to complete the main task.

        ### SUBTASKS ({TaskWorkStatus.COMPLETED.value}):
        These tasks have been reported as completed, and validated as such by the validator; use this section as a reference for progress in the main task.
        ```start_of_completed_subtasks
        {{completed_subtasks}}
        ```end_of_completed_subtasks

        ### SUBTASKS ({TaskWorkStatus.CANCELLED.value}):
        You have previously cancelled these subtasks for various reason and they will not be done.
        ```start_of_cancelled_subtasks
        {{cancelled_subtasks}}
        ```end_of_cancelled_subtasks

        ### SUBTASKS ({TaskWorkStatus.IN_VALIDATION.value}):
        These subtasks have been reported as completed by executors, but are still being validated by validators.
        ```start_of_in_validation_subtasks
        {{in_validation_subtasks}}
        ```end_of_in_validation_subtasks

        ### SUBTASKS ({TaskWorkStatus.IN_PROGRESS.value}):
        These are subtasks that you have delegated to other executors and that are currently being executed by them.
        ```start_of_delegated_subtasks
        {{delegated_subtasks}}
        ```end_of_delegated_subtasks

        ### SUBTASKS ({TaskWorkStatus.BLOCKED.value}):
        These subtasks are blocked by some issue, and execution cannot continue until the issue is resolved, typically by discussing the blocker and/or creating a new subtask to resolve the blocker.
        ```start_of_blocked_subtasks
        {{blocked_subtasks}}
        ```end_of_blocked_subtasks
        """
        return dedent_and_strip(template)

    @property
    def core_state(self) -> CoreState:
        """Overall state of the orchestrator."""
        return CoreState(
            id=self.id,
            knowledge=self.knowledge,
            main_task=self.task,
            subtasks=self.task.subtasks,
            template=self.core_template,
        )

    @property
    def files_dir(self) -> Path:
        """Directory for files related to the orchestrator."""
        return self.files_parent_dir / self.blueprint.id

    @property
    def serialization_location(self) -> Path:
        """Return the location where the orchestrator should be serialized."""
        return self.files_dir / "blueprint.yml"

    @property
    def output_dir(self) -> Path:
        """Output directory of the orchestrator."""
        return self.files_dir / "output"

    @property
    def workspace_dir(self) -> Path:
        """Workspace directory of the orchestrator."""
        return self.files_dir / "workspace"

    @property
    def name(self) -> str:
        """Name of the orchestrator."""
        return self.blueprint.name

    @property
    def auto_wait(self) -> bool:
        """Whether to automatically wait for new events."""
        return self.blueprint.auto_wait

    def make_files_dirs(self) -> None:
        """Make the files directory for the orchestrator."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def serialize(self) -> dict[str, Any]:
        """Serialize the orchestrator to a dict."""
        raise NotImplementedError(
            "TODO: implement serialization of orchestrator itself"
        )
        return asdict(self.blueprint)

    def save(self, update_blueprint: bool = True) -> None:
        """Serialize the orchestrator to YAML."""
        if update_blueprint:
            self.blueprint.rank = self.rank
        # assume that at the point of saving, all executors have been saved and so would have a rank
        assert self.blueprint.rank is not None, "Rank must not be None when saving."
        default_yaml.dump(self.serialize(), self.serialization_location)

    def accepts(self, task: Task) -> bool:
        """Decides whether the orchestrator accepts a task."""
        raise NotImplementedError

    @property
    def recent_events_size(self) -> int:
        """Number of recent events to display."""
        return self.blueprint.recent_events_size

    @property
    def state_update_frequency(self) -> int:
        """How often to update the state of the task, in terms of new events."""
        return max(1, int(self.recent_events_size / 2))

    @property
    def recent_events(self) -> EventLog:
        """Recent events in the event log."""
        return self.task.event_log.recent(self.recent_events_size)

    @property
    def recent_event_status(self) -> str:
        """Status of recent events."""
        template = """
        ## RECENT EVENTS LOG:
        This is a log of recent events that have occurred during the execution of the main task. This is NOT a complete log—use the main task description and subtask statuses to get a complete picture of the current state of the work:
        ```start_of_recent_events_log
        {event_log}
        ```end_of_recent_events_log
        """
        return dedent_and_strip(template).format(
            event_log=self.task.reformat_event_log(
                self.recent_events, pov=Concept.EXECUTOR
            )
        )

    @property
    def default_mode_status(self) -> str:
        """Default status of the orchestrator."""
        template = """
        {core_state}

        {recent_event_status}
        """
        return dedent_and_strip(template).format(
            core_state=str(self.core_state),
            recent_event_status=self.recent_event_status,
        )

    @property
    def default_action_names(self) -> Set[str]:
        """Names of actions available in the default state."""
        return {
            ActionName.IDENTIFY_NEW_SUBTASK.value,
            ActionName.MESSAGE_TASK_OWNER.value,
            ActionName.START_DISCUSSION_FOR_SUBTASK.value,
            ActionName.REPORT_MAIN_TASK_COMPLETE.value,
        }

    @property
    def default_mode_actions(self) -> str:
        """Actions available in the default mode."""
        actions = """
        - `{IDENTIFY_NEW_SUBTASK}`: identify a new subtask from the MAIN TASK that is not yet on the existing subtask list. This adds the subtask to the list and begins a discussion thread with the subtask's executor to start work on the task.
        - `{START_DISCUSSION_FOR_SUBTASK}: "{{id}}"`: open a discussion thread with a subtask's executor, which allows you to exchange information about the subtask. {{id}} must be replaced with the id of the subtask to be discussed.
        - `{MESSAGE_TASK_OWNER}: "{{message}}"`: send a message to the MAIN TASK OWNER to gather or clarify information about the task. {{message}} must be replaced with the message you want to send.
        - `{REPORT_MAIN_TASK_COMPLETE}`: report the main task as complete.
        - `{WAIT}`: do nothing until the next event from an executor or the MAIN TASK OWNER.
        """
        return dedent_and_strip(
            actions.format(
                IDENTIFY_NEW_SUBTASK=ActionName.IDENTIFY_NEW_SUBTASK.value,
                # IDENTIFIED=TaskWorkStatus.IDENTIFIED.value,
                START_DISCUSSION_FOR_SUBTASK=ActionName.START_DISCUSSION_FOR_SUBTASK.value,
                MESSAGE_TASK_OWNER=ActionName.MESSAGE_TASK_OWNER.value,
                REPORT_MAIN_TASK_COMPLETE=ActionName.REPORT_MAIN_TASK_COMPLETE.value,
                WAIT=ActionName.WAIT.value,
            )
        )

    @property
    def action_reasoning_template(self) -> str:
        """Template for action reasoning."""
        template = """
        Use the following reasoning process to decide what to do next:
        ```start_of_reasoning_steps
        {action_choice_core}
        ```end_of_reasoning_steps

        In your reply, you must include output from all steps of the reasoning process, in this block format:
        ```start_of_action_reasoning_output
        1. {{step_1_output}}
        2. {{step_2_output}}
        3. [... etc.]
        ```end_of_action_reasoning_output
        After this block, you must include the action you have decided on, in this format:
        ```start_of_action_choice_output
        justifications: |-
          {{justifications}}
        action_choice: |-
          {{action_choice}} # must be one of the actions listed above, in the same format
        ```end_of_action_choice_output
        Any additional comments or thoughts can be added before or after the output blocks.
        """
        return dedent_and_strip(template)

    @property
    def default_mode_info_sections(self) -> str:
        """Basic information orchestrators in default state have access to."""
        template = f"""
        - {OrchestratorInformationSection.KNOWLEDGE.value}
        - {OrchestratorInformationSection.MAIN_TASK_DESCRIPTION.value}
        - {OrchestratorInformationSection.SUBTASKS.value}
        - {OrchestratorInformationSection.RECENT_EVENTS_LOG.value}
        """
        return dedent_and_strip(template)

    @property
    def subtask_mode_info_sections(self) -> str:
        """Basic information orchestrators in subtask discussion mode have access to."""
        template = f"""
        - {OrchestratorInformationSection.KNOWLEDGE.value}
        - {OrchestratorInformationSection.MAIN_TASK_DESCRIPTION.value}
        - {OrchestratorInformationSection.SUBTASKS.value}
        - {OrchestratorInformationSection.RECENT_EVENTS_LOG.value}
        - {OrchestratorInformationSection.FOCUSED_SUBTASK.value}
        - {OrchestratorInformationSection.FOCUSED_SUBTASK_FULL_DISCUSSION_LOG.value}
        """
        return dedent_and_strip(template)

    @property
    def info_sections(self) -> str:
        """Basic information orchestrators have access to."""
        if self.focused_subtask:
            return self.subtask_mode_info_sections
        return self.default_mode_info_sections

    @property
    def base_info(self) -> str:
        """Basic information orchestrators have access to."""
        template = f"""
        ## CONCEPTS:
        {{orchestrator_concepts}}

        ## {Concept.ORCHESTRATOR_INFORMATION_SECTIONS.value}:
        By default, the orchestrator has access to the following information. Note that all information here is read-only; while identifying new subtasks, the orchestrator cannot modify any of the information here.
        {{orchestrator_information_sections}}
        """
        return dedent_and_strip(template).format(
            orchestrator_concepts=ORCHESTRATOR_CONCEPTS,
            orchestrator_information_sections=self.info_sections,
        )

    @property
    def reasoning_generator(self) -> ReasoningGenerator:
        """Reasoning generator for the orchestrator."""
        return ReasoningGenerator(_orchestrator=self)

    def generate_default_action_reasoning(self) -> str:
        """Generate reasoning for choosing an action in the default state."""
        return self.reasoning_generator.generate_default_action_reasoning()

    @property
    def default_action_reasoning(self) -> str:
        """Prompt for choosing an action in the default mode."""
        if not self.blueprint.reasoning.default_action_choice:
            self.blueprint.reasoning.default_action_choice = (
                self.generate_default_action_reasoning()
            )
        return self.action_reasoning_template.format(
            action_choice_core=self.blueprint.reasoning.default_action_choice,
        )

    @property
    def default_action_context(self) -> str:
        """Prompt for choosing an action in the default state."""
        template = """
        {default_mode_status}

        ## {ORCHESTRATOR_ACTIONS}:
        These are the actions you can currently perform.
        {default_mode_actions}
        """
        return dedent_and_strip(template).format(
            default_mode_status=self.default_mode_status,
            ORCHESTRATOR_ACTIONS=Concept.ORCHESTRATOR_ACTIONS.value,
            default_mode_actions=self.default_mode_actions,
        )

    @property
    def action_mode(self) -> ActionModeName:
        """What action state the orchestrator is in."""
        if self.focused_subtask:
            return ActionModeName.SUBTASK_DISCUSSION
        return ActionModeName.DEFAULT

    @property
    def focused_subtask_discussion(self) -> str:
        """Discussion of the focused subtask."""
        assert self.focused_subtask is not None
        template = """
        ## FOCUSED SUBTASK:
        You are currently focusing on the following subtask:
        ```start_of_subtask_information
        {subtask_information}
        ```end_of_subtask_information

        ### FOCUSED SUBTASK FULL DISCUSSION LOG:
        Below is a complete log of the discussion of the FOCUSED SUBTASK so far. Some messages may overlap with the RECENT EVENTS LOG above, but this log has all messages related to the FOCUSED SUBTASK rather than just the most recent.
        ```start_of_subtask_discussion_log
        {subtask_discussion}
        ```end_of_subtask_discussion_log
        """
        return dedent_and_strip(template).format(
            subtask_information=self.focused_subtask.as_subtask_printout,
            subtask_discussion=self.focused_subtask.reformat_event_log(
                self.focused_subtask.event_log.messages, pov=Concept.MAIN_TASK_OWNER
            ),
        )

    @property
    def subtask_mode_status(self) -> str:
        """Status of the orchestrator in subtask discussion mode."""
        template = """
        {default_mode_status}

        {subtask_discussion}
        """
        assert self.focused_subtask is not None
        return dedent_and_strip(template).format(
            default_mode_status=self.default_mode_status,
            subtask_discussion=self.focused_subtask_discussion,
        )

    @property
    def subtask_mode_actions(self) -> str:
        """Actions available in subtask discussion mode."""
        actions = """
        - `{MESSAGE_TASK_OWNER}: "{{message}}"`: send a message to the {MAIN_TASK_OWNER} to gather or clarify information about the MAIN TASK. `{{message}}` must be replaced with the message you want to send.
        - `{MESSAGE_SUBTASK_EXECUTOR}: "{{message}}"`: send a message to the {EXECUTOR} of the {FOCUSED_SUBTASK} to gather or clarify information about the {FOCUSED_SUBTASK}. {{message}} must be replaced with the message you want to send. _Note_: the {EXECUTOR} is only aware of its own {FOCUSED_SUBTASK}, not _your_ {MAIN_TASK}. From its perspective, the {FOCUSED_SUBTASK} is _its_ {MAIN_TASK}.
        - `{PAUSE_SUBTASK_DISCUSSION}: "{{reason}}"`: pause the discussion of the {FOCUSED_SUBTASK} to either communicate with other subtask executors, the {MAIN_TASK_OWNER}, or to create a new subtask. The {FOCUSED_SUBTASK}'s discussion will be frozen, but can be resumed later. {{reason}} must be replaced with the reason for pausing the discussion, so that the orchestrator can remember why it paused the discussion when it resumes it later.
        - `{CANCEL_SUBTASK}: "{{reason}}"`: cancel the {FOCUSED_SUBTASK} for the given reason. {{reason}} must be replaced with the reason for cancelling the subtask.
        - `{WAIT}`: do nothing until the next event from the {FOCUSED_SUBTASK}.
        """
        return dedent_and_strip(actions).format(
            MAIN_TASK_OWNER=Concept.MAIN_TASK_OWNER.value,
            MAIN_TASK=Concept.MAIN_TASK.value,
            EXECUTOR=Concept.EXECUTOR.value,
            FOCUSED_SUBTASK=Concept.FOCUSED_SUBTASK.value,
            MESSAGE_SUBTASK_EXECUTOR=ActionName.MESSAGE_SUBTASK_EXECUTOR.value,
            MESSAGE_TASK_OWNER=ActionName.MESSAGE_TASK_OWNER.value,
            PAUSE_SUBTASK_DISCUSSION=ActionName.PAUSE_SUBTASK_DISCUSSION.value,
            CANCEL_SUBTASK=ActionName.CANCEL_SUBTASK.value,
            WAIT=ActionName.WAIT.value,
        )

    @property
    def subtask_action_context(self) -> str:
        """Context for choosing an action in subtask discussion mode."""
        template = """
        {subtask_mode_status}

        ## {ORCHESTRATOR_ACTIONS}:
        These are the actions you can currently perform.
        {subtask_mode_actions}
        """
        return dedent_and_strip(template).format(
            subtask_mode_status=self.subtask_mode_status,
            ORCHESTRATOR_ACTIONS=Concept.ORCHESTRATOR_ACTIONS.value,
            subtask_mode_actions=self.subtask_mode_actions,
        )

    @property
    def action_choice_context(self) -> str:
        """Context for choosing an action."""
        if self.focused_subtask:
            return self.subtask_action_context
        if self.action_mode == ActionModeName.DEFAULT:
            return self.default_action_context
        raise NotImplementedError(f"task: {self.task.description}")

    def generate_subtask_action_reasoning(self) -> str:
        """Generate reasoning for choosing an action in subtask discussion mode."""
        return self.reasoning_generator.generate_subtask_action_reasoning()

    @property
    def subtask_action_reasoning(self) -> str:
        """Reasoning for choosing an action in subtask discussion mode."""
        if not self.blueprint.reasoning.subtask_action_choice:
            self.blueprint.reasoning.subtask_action_choice = (
                self.generate_subtask_action_reasoning()
            )
        return self.action_reasoning_template.format(
            action_choice_core=self.blueprint.reasoning.subtask_action_choice,
        )

    @property
    def action_choice_reasoning(self) -> str:
        """Prompt for choosing an action."""
        if self.focused_subtask:
            return self.subtask_action_reasoning
        if self.action_mode == ActionModeName.DEFAULT:
            return self.default_action_reasoning
        raise NotImplementedError

    def choose_action(self) -> ActionDecision:
        """Choose an action to perform."""
        messages = [
            SystemMessage(content=self.action_choice_context),
            SystemMessage(content=self.action_choice_reasoning),
        ]
        action_choice = query_model(
            model=precise_model,
            messages=messages,
            preamble=f"Choosing next action...\n{as_printable(messages)}",
            color=AGENT_COLOR,
        )
        if not (
            extracted_result := extract_blocks(
                action_choice, "start_of_action_choice_output"
            )
        ):
            raise ExtractionError("Could not extract action choice from the result.")
        return ActionDecision.from_yaml_str(extracted_result[0])

    @property
    def event_log(self) -> EventLog:
        """Events that have occurred during the execution of the task."""
        return self.task.event_log

    @property
    def id_generator(self) -> IdGenerator:
        """Id generator for the orchestrator."""
        return self.task.id_generator

    def message_task_owner(self, message: str) -> ActionResult:
        """Send message to main task owner. Main task is blocked until there is a reply."""
        return ActionResult(
            new_events=[
                Event(
                    data=Message(
                        sender=self.id, recipient=self.task.owner_id, content=message
                    ),
                    generating_task_id=self.task.id,
                    id=generate_swarm_id(EventId, self.id_generator),
                ),
            ],
            pause_execution=PauseExecution(True),
            new_work_status=TaskWorkStatus.BLOCKED,
        )

    @property
    def subtask_extraction_context(self) -> str:
        """Context for extracting a subtask."""
        template = """
        {default_status}

        ## MODULAR SUBTASK IDENTIFICATION PHILOSOPHY (MSI):
        {msi}
        """
        return dedent_and_strip(template).format(
            default_status=self.default_mode_status,
            msi=MODULAR_SUBTASK_IDENTIFICATION,
        )

    def generate_subtask_identification_reasoning(self) -> str:
        """Generate reasoning for extracting a subtask."""
        return self.reasoning_generator.generate_subtask_identification_reasoning()

    @property
    def subtask_extraction_reasoning(self) -> str:
        """Prompt for extracting a subtask."""
        if not self.blueprint.reasoning.subtask_extraction:
            self.blueprint.reasoning.subtask_extraction = (
                self.generate_subtask_identification_reasoning()
            )
        template = """
        Use the following reasoning process to decide what to do next:
        ```start_of_reasoning_steps
        {subtask_extraction_core}
        ```end_of_reasoning_steps

        In your reply, you must include output from all steps of the reasoning process, in this block format:
        ```start_of_reasoning_output
        1. {{step_1_output}}
        2. {{step_2_output}}
        3. [... etc.]
        ```end_of_reasoning_output
        After this block, you must include the subtask you have identified for its executor. To the executor, the identified subtask becomes its own MAIN TASK, and you are the MAIN TASK OWNER of the subtask. The executor knows nothing about your original MAIN TASK. The subtask must be described in the following format:
        ```start_of_subtask_identification_output
        identified_subtask: |- # high-level, single-sentence description of the subtask
          {{identified_subtask}}
        ```end_of_subtask_identification_output
        Any additional comments or thoughts can be added before or after the output blocks.
        """
        return dedent_and_strip(template).format(
            subtask_extraction_core=self.blueprint.reasoning.subtask_extraction
        )

    @property
    def subtasks(self) -> TaskList:
        """Subtasks of the orchestrator."""
        return self.task.subtasks

    @property
    def validator_state(self) -> str:
        """State sent to the validator."""
        template = """
        ## MAIN TASK DESCRIPTION:
        Here is information about the main task being worked on:
        ```start_of_main_task_description
        {task_specification}
        ```end_of_main_task_description

        ## SUBTASKS:
        Here are the subtasks that have been identified so far:

        ### SUBTASKS (COMPLETED):
        ```start_of_completed_subtasks
        {completed_subtasks}
        ```end_of_completed_subtasks

        ### SUBTASKS (CANCELLED):
        ```start_of_cancelled_subtasks
        {cancelled_subtasks}
        ```end_of_cancelled_subtasks

        ### SUBTASKS (IN_VALIDATION):
        ```start_of_in_validation_subtasks
        {in_validation_subtasks}
        ```end_of_in_validation_subtasks

        ### SUBTASKS (IN_PROGRESS):
        ```start_of_delegated_subtasks
        {delegated_subtasks}
        ```end_of_delegated_subtasks

        ### SUBTASKS (BLOCKED):
        ```start_of_blocked_subtasks
        {blocked_subtasks}
        ```end_of_blocked_subtasks
        """

        completed_subtasks = str(
            self.subtasks.filter_by_status(TaskWorkStatus.COMPLETED)
        )
        cancelled_subtasks = str(
            self.subtasks.filter_by_status(TaskWorkStatus.CANCELLED)
        )
        in_validation_subtasks = str(
            self.subtasks.filter_by_status(TaskWorkStatus.IN_VALIDATION)
        )
        delegated_subtasks = str(
            self.subtasks.filter_by_status(TaskWorkStatus.IN_PROGRESS)
        )
        # new_subtasks = str(self.subtasks.filter_by_status(TaskWorkStatus.IDENTIFIED))
        blocked_subtasks = str(self.subtasks.filter_by_status(TaskWorkStatus.BLOCKED))
        return dedent_and_strip(template).format(
            task_specification=str(self.task),
            completed_subtasks=completed_subtasks,
            cancelled_subtasks=cancelled_subtasks,
            in_validation_subtasks=in_validation_subtasks,
            delegated_subtasks=delegated_subtasks,
            # new_subtasks=new_subtasks,
            blocked_subtasks=blocked_subtasks,
        )

    def validate_subtask_identification(self, subtask: str) -> ValidationResult:
        """Validate some work."""
        instructions = """
        {validator_state}

        ## REQUEST FOR YOU:
        Please check that the subtask identification is correct:
        - Subtask: {subtask}
        """
        instructions = dedent_and_strip(instructions).format(
            validator_state=self.validator_state,
            subtask=subtask,
        )
        return self.task.validator.validate(instructions)

    def add_subtask(self, subtask: Task) -> None:
        """Add a subtask to the orchestrator."""
        self.task.subtasks.items.append(subtask)

    def subtask_message(self, subtask: Task, message: str) -> Event:
        """Format a message to a subtask."""
        assert (
            subtask.executor_id is not None
        ), "Cannot post message to subtask without an executor."
        return Event(
            data=Message(
                sender=self.id,
                recipient=subtask.executor_id,
                content=message,
            ),
            generating_task_id=subtask.id,
            id=generate_swarm_id(EventId, self.id_generator),
        )

    def send_subtask_message(
        self, message_text: str, initial: bool = False
    ) -> list[Event]:
        """Send a message to the executor for the focused subtask."""
        assert (focused_subtask := self.focused_subtask) is not None
        message_event = self.subtask_message(focused_subtask, message_text)
        focused_subtask.event_log.add(message_event)
        report_status_change = (
            not initial and focused_subtask.work_status != TaskWorkStatus.IN_PROGRESS
        )
        focused_subtask.work_status = TaskWorkStatus.IN_PROGRESS
        status_change_event = Event(
            data=TaskStatusChange(
                changing_agent=self.id,
                task_id=focused_subtask.id,
                old_status=focused_subtask.work_status,
                new_status=TaskWorkStatus.IN_PROGRESS,
                reason=f"Sent message to {Concept.EXECUTOR.value} regarding subtask.",
            ),
            generating_task_id=self.task.id,
            id=generate_swarm_id(EventId, self.id_generator),
        )
        return [status_change_event] if report_status_change else []

    def focus_subtask(self, subtask: Task) -> Event:
        """Focus on a subtask."""
        self.state.focused_subtask = subtask
        return Event(
            data=SubtaskFocus(
                owner_id=self.id,
                subtask_id=subtask.id,
            ),
            generating_task_id=self.task.id,
            id=generate_swarm_id(EventId, self.id_generator),
        )

    def identify_new_subtask(self) -> ActionResult:
        """Identify a new subtask."""
        messages = [
            SystemMessage(content=self.subtask_extraction_context),
            SystemMessage(content=self.subtask_extraction_reasoning),
        ]
        new_subtask = query_model(
            model=precise_model,
            messages=messages,
            preamble=f"Extracting subtask...\n{as_printable(messages)}",
            color=AGENT_COLOR,
        )
        extracted_results = extract_blocks(
            new_subtask, "start_of_subtask_identification_output"
        )
        if not extracted_results:
            raise ExtractionError(
                f"Could not extract subtask from the result:\n{new_subtask}"
            )
        extracted_results = SubtaskIdentifcationResult.from_yaml_str(
            extracted_results[-1]
        )
        identified_subtask = extracted_results.identified_subtask
        subtask = Task(
            name=identified_subtask,
            owner_id=self.id,
            rank_limit=None if self.rank_limit is None else self.rank_limit - 1,
            description=TaskDescription(information=identified_subtask),
            validator=self.task.validator,
            id_generator=self.id_generator,
        )
        subtask_validation = self.validate_subtask_identification(identified_subtask)
        subtask_identification_event = Event(
            data=SubtaskIdentification(
                owner_id=self.id,
                subtask=identified_subtask,
                subtask_id=subtask.id,
                validation_result=subtask_validation,
            ),
            generating_task_id=self.task.id,
            id=generate_swarm_id(EventId, self.id_generator),
        )
        additional_thoughts_event = (
            Event(
                data=Thought(
                    agent_id=self.id,
                    content=extracted_results.additional_thoughts,
                ),
                generating_task_id=self.task.id,
                id=generate_swarm_id(EventId, self.id_generator),
            )
            if extracted_results.additional_thoughts
            else None
        )
        if not subtask_validation.valid:
            return ActionResult(
                pause_execution=PauseExecution(False),
                new_events=[subtask_identification_event],
            )
        self.delegator.assign_executor(subtask, self.recent_events_size, self.auto_wait)
        assert subtask.executor is not None, "Task executor assignment failed."
        self.add_subtask(subtask)
        subtask_focus_event = self.focus_subtask(subtask)
        subtask_initiation_events = self.send_subtask_message(
            message_text="Hi, please feel free to ask me any questions about the context of this task—I've only given you a brief description to start with, but I can provide more information if you need it.",
            initial=True,
        )
        new_events = [
            subtask_identification_event,
            additional_thoughts_event,
            subtask_focus_event,
            *subtask_initiation_events,
        ]
        new_events = [event for event in new_events if event is not None]
        return ActionResult(
            pause_execution=PauseExecution(False),
            new_events=new_events,
        )

    @property
    def awaitable_subtasks(self) -> TaskList:
        """Subtasks that can be awaited."""
        # if in default mode, awaitable subtasks are all in-progress subtasks
        if self.focused_subtask:
            return (
                TaskList([self.focused_subtask])
                if self.focused_subtask.work_status == TaskWorkStatus.IN_PROGRESS
                else TaskList()
            )
        if self.action_mode == ActionModeName.DEFAULT:
            return self.subtasks.filter_by_status(TaskWorkStatus.IN_PROGRESS)
        raise NotImplementedError

    @property
    def subtask_action_names(self) -> Set[str]:
        """Names of actions available in subtask discussion mode."""
        return {
            ActionName.MESSAGE_SUBTASK_EXECUTOR.value,
            ActionName.PAUSE_SUBTASK_DISCUSSION.value,
            ActionName.MESSAGE_TASK_OWNER.value,
            ActionName.WAIT.value,
        }

    @property
    def action_names(self) -> Set[str]:
        """Names of actions available to the orchestrator."""
        if self.focused_subtask:
            return self.subtask_action_names
        return self.default_action_names

    def message_subtask_executor(self, message: str) -> ActionResult:
        """Send message to subtask executor."""
        return ActionResult(
            new_events=self.send_subtask_message(message),
            pause_execution=PauseExecution(False),
            new_work_status=TaskWorkStatus.IN_PROGRESS,
        )

    def act(self, decision: ActionDecision) -> ActionResult:
        """Act on a decision."""
        decision.validate_action(valid_actions=self.action_names)
        if decision.action_name == ActionName.MESSAGE_TASK_OWNER.value:
            return self.message_task_owner(decision.action_args["message"])
        if decision.action_name == ActionName.IDENTIFY_NEW_SUBTASK.value:
            return self.identify_new_subtask()
        if decision.action_name == ActionName.START_DISCUSSION_FOR_SUBTASK.value:
            raise NotImplementedError
        if decision.action_name == ActionName.REPORT_MAIN_TASK_COMPLETE.value:
            raise NotImplementedError
        if decision.action_name == ActionName.WAIT.value:
            raise NotImplementedError
        if decision.action_name == ActionName.MESSAGE_SUBTASK_EXECUTOR.value:
            return self.message_subtask_executor(decision.action_args["message"])
        if decision.action_name == ActionName.PAUSE_SUBTASK_DISCUSSION.value:
            raise NotImplementedError

        # > change task cancellation to task failing validation
        # > if a task is set to be complete, trigger validation agent automatically
        # > need to add fail reason for failed tasks
        # ....
        # (next_action_implementation) > pause subtask discussion: adds event that is a summary of the new items in the discussion to maintain state continuity # "The Definition of Done is a Python script that, when run, starts the agent. The agent should be able to have a simple back-and-forth conversation with the user. The agent needs to use the OpenAI Assistant API."
        raise NotImplementedError

    def message_from_owner(self, message: str) -> Event:
        """Create a message from the task owner."""
        return Event(
            data=Message(
                sender=self.task.owner_id,
                recipient=self.id,
                content=message,
            ),
            generating_task_id=self.task.id,
            id=generate_swarm_id(EventId, self.id_generator),
        )

    def message_to_owner(self, message: str) -> Event:
        """Create a message to the task owner."""
        return Event(
            data=Message(
                sender=self.id,
                recipient=self.task.owner_id,
                content=message,
            ),
            generating_task_id=self.task.id,
            id=generate_swarm_id(EventId, self.id_generator),
        )

    @property
    def first_new_event(self) -> Event:
        """First new event since the last update of the main task."""
        return self.event_log.events[-(self.new_event_count)]

    @property
    def last_read_message(self) -> Event | None:
        """Last message read by the orchestrator."""
        old_events = reversed(self.event_log.events[: -(self.new_event_count)])
        old_messages = (
            event for event in old_events if isinstance(event.data, Message)
        )
        return next(old_messages, None)

    def generate_main_task_description_update(
        self,
    ) -> tuple[TaskDescription, str | None] | None:
        """Generate an update to the main task description."""
        reasoning = f"""
        1. Review the {Concept.MAIN_TASK_INFORMATION.value} and the {Concept.MAIN_TASK_DEFINITION_OF_DONE.value} to recall the current status and objectives of the {Concept.MAIN_TASK.value}. Note any specific requirements or key details that may be affected by new information.
        2. Check the {Concept.LAST_READ_MAIN_TASK_OWNER_MESSAGE.value} to identify where in the {Concept.TASK_MESSAGES.value} section you will begin integrating new information. The messages that come after this will hold the updates you need to consider.
        3. Sequentially read and analyze the messages in the {Concept.TASK_MESSAGES.value} section that follow after the {Concept.LAST_READ_MAIN_TASK_OWNER_MESSAGE.value}. For each message:
           a. Determine whether the message contains new information or requests that change the nature or the details of the {Concept.MAIN_TASK.value}.
           b. Evaluate if the new information influences the completion criteria outlined in the {Concept.MAIN_TASK_DEFINITION_OF_DONE.value}.
           c. Note any information that requires clarification or follow-up from the {Concept.MAIN_TASK_OWNER.value} and formulate a message to send if necessary.
        4. Synthesize the new information from step 3 into a concise summary, highlighting the changes impacting the {Concept.MAIN_TASK.value}. Record how the {Concept.MAIN_TASK_INFORMATION.value} or {Concept.MAIN_TASK_DEFINITION_OF_DONE.value} may need to be adjusted based on this information.
        5. Review the synthesized summary and establish a mental update plan by correlating the necessary changes with either the {Concept.MAIN_TASK_INFORMATION.value} or {Concept.MAIN_TASK_DEFINITION_OF_DONE.value} sections.
           a. For each section, list out the points that will be updated.
           b. Organize these points to ensure they are logical and do not conflict with each other.
        """
        reasoning = dedent_and_strip(reasoning)

        # You are a task specification agent that specializes in understanding the context for a {Concepts.MAIN_TASK.value} and updating its information sections based on new information.
        context = f"""
        ## MISSION:
        You are an advanced task orchestrator that specializes in managing the execution of a {Concept.MAIN_TASK.value} and delegating its {Concept.SUBTASK.value} to an {Concept.EXECUTOR.value} that can execute those tasks, while communicating with the {Concept.MAIN_TASK_OWNER.value} to gather required information for the {Concept.MAIN_TASK.value}.

        ### MODE:
        Currently you are NOT communicating with the {Concept.MAIN_TASK_OWNER.value}, but reviewing recent communications with them to update the {Concept.MAIN_TASK_INFORMATION.value} and {Concept.MAIN_TASK_DEFINITION_OF_DONE.value} sections based on new information.

        ## {Concept.MAIN_TASK_INFORMATION.value}:
        {{main_task_information}}

        ## {Concept.MAIN_TASK_DEFINITION_OF_DONE.value}:
        {{main_task_dod}}

        ## {Concept.TASK_MESSAGES.value}
        Here are the messages with the {Concept.MAIN_TASK_OWNER.value}:
        ```start_of_task_messages
        {{recent_messages}}
        ```end_of_task_messages
        {Concept.LAST_READ_MAIN_TASK_OWNER_MESSAGE.value}: {{last_read_main_task_owner_message}}
        """
        context = dedent_and_strip(context).format(
            main_task_information=self.task.information,
            main_task_dod=self.task.definition_of_done,
            recent_messages=self.task.reformat_event_log(
                self.task.messages, pov=Concept.EXECUTOR
            ),
            last_read_main_task_owner_message=self.last_read_message
            or "No messages read.",
        )
        task = """
        ## REQUEST FOR YOU:
        Use the following reasoning process to determine what must be updated in the {MAIN_TASK_DESCRIPTION} and {MAIN_TASK_DEFINITION_OF_DONE} sections:
        ```start_of_reasoning_steps
        {reasoning_steps}
        ```end_of_reasoning_steps

        {reasoning_output_instructions}

        After this block, if the reasoning process determined that there is new information about the {MAIN_TASK}, use the information from the reasoning process to rewrite the {MAIN_TASK_DESCRIPTION} and {MAIN_TASK_DEFINITION_OF_DONE} sections to reflect the new information, in this format:
        ```start_of_main_task_info
        main_task_information: |-
          {{updated main task information}}
        main_task_definition_of_done:
          - {{requirement 1}}
          - {{requirement 2}}
          - [... etc.]
        ```end_of_main_task_info

        If there is no new information about the {MAIN_TASK}, then return the following:
        ```start_of_main_task_info
        NO NEW INFORMATION
        ```end_of_main_task_info
        """
        task = dedent_and_strip(task).format(
            MAIN_TASK=Concept.MAIN_TASK.value,
            MAIN_TASK_DESCRIPTION=Concept.MAIN_TASK_INFORMATION.value,
            MAIN_TASK_DEFINITION_OF_DONE=Concept.MAIN_TASK_DEFINITION_OF_DONE.value,
            reasoning_steps=reasoning,
            reasoning_output_instructions=REASONING_OUTPUT_INSTRUCTIONS,
        )
        messages = [
            SystemMessage(content=context),
            SystemMessage(content=task),
        ]
        result = query_model(
            model=precise_model,
            messages=messages,
            preamble=f"Updating main task description...\n{as_printable(messages)}",
            color=AGENT_COLOR,
        )
        extracted_result = extract_blocks(result, "start_of_main_task_info")
        if not extracted_result:
            raise ExtractionError(
                f"Could not extract main task description from the result:\n{result}"
            )
        if "NO NEW INFORMATION" in extracted_result[-1]:
            return None
        try:
            extracted_result = default_yaml.load(extracted_result[-1])
        except YAMLError as error:
            raise ExtractionError(
                f"Could not extract main task description dictionary from the result:\n{result}"
            ) from error
        return (
            TaskDescription(
                information=extracted_result["main_task_information"],
                definition_of_done=as_yaml_str(
                    extracted_result["main_task_definition_of_done"], YAML()
                ),
            ),
            None
            if (followup_needed := extracted_result.get("needed_followup"))
            and "NONE" in followup_needed
            else followup_needed,
        )

    def update_main_task_description(self) -> None:
        """Update the main task from new events."""
        update_result = self.generate_main_task_description_update()
        if update_result is None:
            return
        (
            updated_task_description,
            followup_needed,
        ) = update_result
        task_update_event = Event(
            data=TaskDescriptionUpdate(
                changing_agent=self.id,
                task_id=self.task.id,
                old_description=str(self.task.description),
                new_description=str(updated_task_description),
                reason=f"new information from latest events in {Concept.RECENT_EVENTS_LOG.value}",
            ),
            generating_task_id=self.task.id,
            id=generate_swarm_id(EventId, self.id_generator),
        )
        followup_event = (
            None
            if followup_needed is None
            else Event(
                data=Thought(
                    agent_id=self.id,
                    content=followup_needed,
                ),
                generating_task_id=self.task.id,
                id=generate_swarm_id(EventId, self.id_generator),
            )
        )
        self.task.description = updated_task_description
        self.event_log.add(task_update_event)
        if followup_event is not None:
            self.event_log.add(followup_event)

    def add_to_event_log(self, events: Sequence[Event]) -> None:
        """Add events to the event log."""
        self.task.event_log.add(*events)
        self.state.new_event_count += len(events)
        if self.new_event_count >= self.state_update_frequency:
            self.update_main_task_description()
            self.state.new_event_count = 0

    def add_thought(self, thought: str) -> None:
        """Add a thought to the event log."""
        self.add_to_event_log(
            [
                Event(
                    data=Thought(
                        agent_id=self.id,
                        content=thought,
                    ),
                    generating_task_id=self.task.id,
                    id=generate_swarm_id(EventId, self.id_generator),
                )
            ]
        )

    async def execute(self, message: str | None = None) -> ExecutorReport:
        """Execute the task. Adds a message (if provided) to the task's event log, and adds own message to the event log at the end of execution."""
        if message is not None:
            self.add_to_event_log([self.message_from_owner(message)])
        old_work_status = self.task.work_status
        while True:
            if self.auto_wait and self.awaitable_subtasks:
                executor_reports = [
                    asyncio.create_task(subtask.executor.execute())
                    for subtask in self.awaitable_subtasks
                    if subtask.executor is not None
                ]
                done_reports, _ = await asyncio.wait(
                    executor_reports, return_when=asyncio.FIRST_COMPLETED
                )
                executor_events = list(
                    chain(
                        *[
                            report.result().events
                            for report in done_reports
                            if report.result().events
                        ]
                    )
                )
                executor_events.sort(key=lambda event: event.timestamp)
                self.add_to_event_log(executor_events)
            action_decision = self.choose_action()
            if action_decision.additional_thoughts:
                self.add_thought(action_decision.additional_thoughts)
            action_result = self.act(action_decision)
            if action_result.new_events:
                self.add_to_event_log(action_result.new_events)
            if action_result.new_work_status:
                self.task.work_status = action_result.new_work_status
            if action_result.pause_execution:
                break
        if not (last_event := self.task.event_log.last_event):
            raise NotImplementedError
        if not isinstance(last_event.data, Message):  # type: ignore
            raise NotImplementedError
        if (
            last_event.data.sender != self.id
            and last_event.data.recipient != self.task.owner_id
        ):
            raise NotImplementedError
        if self.task.work_status != TaskWorkStatus.BLOCKED:
            raise NotImplementedError
        status_change_reason = "Task is blocked until reply to message."
        events = (
            [
                Event(
                    data=TaskStatusChange(
                        changing_agent=self.id,
                        task_id=self.task.id,
                        old_status=old_work_status,
                        new_status=self.task.work_status,
                        reason=status_change_reason,
                    ),
                    generating_task_id=self.task.id,
                    id=generate_swarm_id(EventId, self.id_generator),
                )
            ]
            if self.task.work_status != old_work_status
            else []
        )
        return ExecutorReport(
            reply=last_event.data.content,
            events=events,
        )

    @classmethod
    def load(
        cls,
        blueprint_location: Path,
        task: Task,
        files_parent_dir: Path,
        delegator: "Delegator",
    ) -> Self:
        """Deserialize an orchestrator from a YAML file."""
        blueprint_data = default_yaml.load(blueprint_location)
        blueprint_data["task_history"] = tuple(blueprint_data["task_history"])
        return cls(
            blueprint=OrchestratorBlueprint(**blueprint_data),
            task=task,
            files_parent_dir=files_parent_dir,
            delegator=delegator,
        )


@dataclass
class Reply:
    """A reply from the main agent."""

    content: str
    continue_func: Callable[[str], Coroutine[Any, Any, str]]

    async def continue_conversation(self, message: str) -> str:
        """Continue the conversation with a message."""
        return await self.continue_func(message)


class ExecutorLoader(Protocol):
    """A loader of an executor."""

    def __call__(self, blueprint: Blueprint, task: Task, files_dir: Path) -> Executor:
        """Load an executor."""
        raise NotImplementedError


def extract_bot_loader(loader_location: Path) -> ExecutorLoader:
    """Extract a bot loader from a loader file."""
    assert loader_location.exists(), f"Loader not found: {loader_location}"
    module_name = loader_location.stem
    spec = importlib.util.spec_from_file_location(module_name, loader_location)
    assert spec and spec.loader, f"Could not load loader: {loader_location}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    assert hasattr(module, "load_bot"), "'load_bot' not found in the module"
    return getattr(module, "load_bot")


def load_executor(blueprint: Blueprint, task: Task, files_dir: Path) -> Executor:
    """Factory function for loading an executor from a blueprint."""
    if blueprint.role == Role.BOT:
        loader_location = files_dir / "loader.py"
        load_bot = extract_bot_loader(loader_location)
        return load_bot(blueprint, task, files_dir)
    raise NotImplementedError


class Advisor(Protocol):
    """A single-reply advisor for some issue."""

    def advise(self, prompt: str) -> str:
        """Advise on some issue."""
        raise NotImplementedError


def get_choice(prompt: str, allowed_choices: Set[Any], advisor: Advisor) -> Any:
    """Get a choice from the advisor."""
    while True:
        if (choice := advisor.advise(prompt)) in allowed_choices:
            return choice
        prompt = f"Invalid input. Valid choices: {allowed_choices}."


@dataclass
class BlueprintSearchResult:
    """Result of a blueprint search."""

    blueprint: Blueprint
    task_subpool: list[Task] = field(default_factory=list)

    @property
    def success_rate(self) -> float | None:
        """Success rate of the blueprint given the tasks."""
        if not self.task_subpool:
            return None
        raise NotImplementedError
        # success_rate = (
        #     num_success := sum(task.success for task in self.task_pool)
        # ) / (num_similar_tasks := len(self.task_pool))

    @property
    def task_subpool_size(self) -> int:
        """Number of similar tasks."""
        return len(self.task_subpool)

    @property
    def completion_time(self) -> float | None:
        """Completion time of the blueprint given the tasks."""
        if not self.task_subpool:
            return None
        raise NotImplementedError
        # completion_time = (
        #     sum(task.completion_time for task in self.task_pool)
        #     / task_pool_size
        # )

    @property
    def scaled_completion_time(self) -> float | None:
        """Scaled completion time of the blueprint given the tasks."""
        if not self.task_subpool:
            return None
        raise NotImplementedError
        # scaled_completion_time = completion_time / (1 + completion_time)

    @property
    def rating(self) -> float | None:
        """Rating of the blueprint given the tasks."""
        if not self.task_subpool:
            return None
        raise NotImplementedError
        # rating = success_rate / (1 + scaled_completion_time)


DelegationSuccessful = NewType("DelegationSuccessful", bool)


def find_similar_tasks(task: Task, task_pool: Iterable[Task]) -> list[Task]:
    """Find similar tasks in a task pool."""
    if not task_pool:
        return []
    raise NotImplementedError


def search_task_records(task: Task, task_records_dir: Path) -> list[Task]:
    """Search for similar tasks in the task records."""
    if not list(task_records_dir.iterdir()):
        return []
    raise NotImplementedError


def rerank_tasks(task: Task, similar_tasks: list[Task]) -> list[Task]:
    """Rerank similar tasks based on task similarity."""
    raise NotImplementedError


def is_bot(blueprint: Blueprint) -> bool:
    """Check if a blueprint is a bot."""
    return blueprint.rank == 0


def load_blueprint(blueprint_path: Path) -> Blueprint:
    """Load a blueprint from a file."""
    blueprint_data = default_yaml.load(blueprint_path)
    role = Role(blueprint_data["role"])
    if role == Role.BOT:
        return BotBlueprint.deserialize(blueprint_data)
    raise NotImplementedError


def load_blueprints(executors_dir: Path) -> Iterable[Blueprint]:
    """Load blueprints from the executors directory."""
    dirs = (
        executor_dir
        for executor_dir in executors_dir.iterdir()
        if executor_dir.is_dir()
    )

    return (load_blueprint(executor_dir / "blueprint.yaml") for executor_dir in dirs)


def is_new(blueprint: Blueprint, task_history_limit: int = 10) -> bool:
    """Check if a blueprint is new."""
    return len(blueprint.task_history) <= task_history_limit


def load_tasks(task_ids: Iterable[TaskId], task_records_dir: Path) -> list[Task]:
    """Load tasks from the task records."""
    if not task_ids:
        return []
    raise NotImplementedError


@dataclass(frozen=True)
class Delegator:
    """Delegates tasks to executors, creating new ones if needed."""

    executors_dir: Path
    task_records_dir: Path
    task_search_rerank_threshold: int
    id_generator: IdGenerator

    @cached_property
    def id(self) -> DelegatorId:
        """Id of the delegator."""
        return generate_swarm_id(DelegatorId, self.id_generator)

    def search_blueprints(
        self,
        task: Task,
        rank_limit: int | None = None,
    ) -> list[BlueprintSearchResult]:
        """Search for blueprints of executors that can handle a task."""
        similar_tasks = search_task_records(task, self.task_records_dir)
        if len(similar_tasks) > self.task_search_rerank_threshold:
            similar_tasks = rerank_tasks(task, similar_tasks)
        past_blueprint_ids = [task.executor_blueprint_id for task in similar_tasks]

        def is_candidate(blueprint: Blueprint) -> bool:
            """Check if a blueprint is a candidate for the task."""
            assert blueprint.rank is not None
            if (
                blueprint.id not in past_blueprint_ids
                and not is_bot(blueprint)
                or (rank_limit is not None and blueprint.rank > rank_limit)
            ):
                return False
            if is_new(blueprint):
                return True

            raise NotImplementedError
            # > TODO: filter by minimum success rate, given large enough task history > task success is restricted to similar tasks that executor dealt with before
            # > need to be able to exclude bots as normal based on success rate—they might not be suitable for the task

        candidate_blueprints = [
            blueprint
            for blueprint in load_blueprints(self.executors_dir)
            if is_candidate(blueprint)
        ]
        if not candidate_blueprints:
            return []
        search_results: list[BlueprintSearchResult] = []
        for blueprint in candidate_blueprints:
            blueprint_tasks = load_tasks(blueprint.task_history, self.task_records_dir)
            candidate_similar_tasks = find_similar_tasks(task, blueprint_tasks)
            if not candidate_similar_tasks:
                assert is_bot(
                    blueprint
                ), f"Blueprint search: no similar tasks found for non-bot blueprint:\n\nBlueprint:\n{blueprint}\n\nTask:\n{task}"
            search_results.append(
                BlueprintSearchResult(blueprint, candidate_similar_tasks)
            )
        return search_results

    def choose_next_executor(
        self,
        candidates: list[BlueprintSearchResult],
        task: Task,
    ) -> BlueprintSearchResult:
        """Evaluate candidates for a task."""
        raise NotImplementedError
        # TODO: use executor selection reasoning to choose next agent to ask

    def make_executor(
        self, task: Task, recent_events_size: int, auto_await: bool
    ) -> Executor:
        """Factory for creating a new executor for a task."""
        if task.rank_limit and task.rank_limit > 0:
            raise NotImplementedError(
                "Unable to automatically create 0-ranked executor."
            )
        blueprint = OrchestratorBlueprint(
            name=f"orchestrator_{task.id}",
            rank=None,
            task_history=[task.id],
            reasoning=Reasoning(),
            knowledge="",
            recent_events_size=recent_events_size,
            auto_wait=auto_await,
            id=generate_swarm_id(BlueprintId, self.id_generator),
        )
        return Orchestrator(
            blueprint=blueprint,
            task=task,
            files_parent_dir=self.executors_dir,
            delegator=self,
        )

    def find_top_candidates(
        self,
        candidates: Sequence[BlueprintSearchResult],
        max_candidates: int,
    ) -> list[BlueprintSearchResult]:
        """Find the top candidates."""
        candidates = sorted(
            candidates,
            key=lambda result: float("inf")
            if (rating := result.rating is None)
            else rating,
            reverse=True,
        )
        if len(candidates) <= max_candidates:
            return candidates
        raise NotImplementedError
        # TODO: separate list for new vs old executors; take max 1/3 from new, max 2/3 from old, fill rest with remaining list

    def reorder_candidate_list(
        self,
        candidates: list[BlueprintSearchResult],
        task: Task,
    ) -> Generator[BlueprintSearchResult, None, None]:
        """Reorder the candidate list."""
        chosen: set[BlueprintId] = set()
        while len(chosen) < len(candidates):
            available_candidates = [
                candidate
                for candidate in candidates
                if candidate.blueprint.id not in chosen
            ]
            if len(available_candidates) == 1:
                yield available_candidates[0]
            next_candidate = self.choose_next_executor(available_candidates, task)
            chosen.add(next_candidate.blueprint.id)
            yield next_candidate

    def delegate(
        self,
        task: Task,
        max_candidates: int = 10,
    ) -> DelegationSuccessful:
        """Find an executor to delegate the task to."""
        candidates = self.search_blueprints(task, task.rank_limit)
        if not candidates:
            return DelegationSuccessful(False)
        candidates = self.find_top_candidates(candidates, max_candidates)
        for candidate in self.reorder_candidate_list(candidates, task):
            candidate = load_executor(
                candidate.blueprint, task, self.executors_dir / candidate.blueprint.id
            )
            if candidate.accepts(task):
                task.executor = candidate
                task.rank_limit = candidate.rank
                return DelegationSuccessful(True)
            raise NotImplementedError
            # TODO: refusal of agent counts as failure in task history
        return DelegationSuccessful(False)

    def assign_executor(
        self, task: Task, recent_events_size: int, auto_await: bool
    ) -> None:
        """Assign an existing or new executor to a task."""
        delegation_successful = self.delegate(task)
        # blueprints represent known capabilities; so, failure means we must create a new executor
        if not delegation_successful:
            task.executor = self.make_executor(task, recent_events_size, auto_await)


@dataclass
class Swarm:
    """Main interfacing class for the agent."""

    files_dir: Path = Path(".data")
    """Directory for files related to the agent and any subagents."""
    work_validator: WorkValidator = field(
        default_factory=lambda: Human(name="Human Validator")
    )
    """Agent that approves or rejects work."""
    recent_events_size: int = 15
    """Number of recent events to display in orchestrators' event logs."""
    auto_wait: bool = True
    """Whether orchestrators will automatically wait for their executors. If disabled, orchestrators may perform other actions while an executor works on a task."""
    task_search_rerank_threshold: int = 100
    """When searching for similar past tasks, run a reranker if there are more than this many tasks."""
    id_generator: IdGenerator = field(default_factory=DefaultIdGenerator)
    """Generator for ids of entities in the system."""
    llm_cache_enabled: InitVar[bool] = field(default=True)
    """Whether to enable the LLM cache for identical calls to models."""

    def __post_init__(self, llm_cache_enabled: bool) -> None:
        """Post-initialization hook."""
        if llm_cache_enabled:
            configure_langchain_cache()

    @property
    def cache_dir(self) -> Path:
        """Directory for the LLM cache."""
        if not (cache_dir := self.files_dir / ".cache").exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    @property
    def executors_dir(self):
        """Directory for executors."""
        if not (executors_dir := self.files_dir / "executors").exists():
            executors_dir.mkdir(parents=True, exist_ok=True)
        return executors_dir

    @property
    def task_records_dir(self):
        """Directory for task records."""
        if not (task_records_dir := self.files_dir / "task_records").exists():
            task_records_dir.mkdir(parents=True, exist_ok=True)
        return task_records_dir

    @cached_property
    def delegator(self) -> Delegator:
        """Delegator for assigning tasks to executors."""
        return Delegator(
            executors_dir=self.executors_dir,
            task_records_dir=self.task_records_dir,
            task_search_rerank_threshold=self.task_search_rerank_threshold,
            id_generator=self.id_generator,
        )

    @cached_property
    def id(self) -> RuntimeId:
        """Runtime id of the agent."""
        return RuntimeId(str(self.id_generator()))

    @property
    def name(self) -> str:
        """Name of the agent."""
        return f"swarm_{self.id}"

    async def run(self, message: str) -> Reply:
        """Run the agent with a message, and a way to continue the conversation. Rerunning this method starts a new conversation."""
        task = Task(
            description=TaskDescription(information=message),
            owner_id=self.id,
            rank_limit=None,
            validator=self.work_validator,
            id_generator=self.id_generator,
        )
        self.delegator.assign_executor(task, self.recent_events_size, self.auto_wait)
        assert task.executor is not None, "Task executor assignment failed."
        executor_report = await task.executor.execute()
        breakpoint()

        async def continue_conversation(message: str) -> str:
            """Continue the conversation with a message."""
            assert (
                task.executor is not None
            ), "Task executor must exist in order to be executed."
            return (await task.executor.execute(message)).reply

        return Reply(
            content=executor_report.reply,
            continue_func=continue_conversation,
        )


# > execution needs to not have a parameter
# > when initally agent is saved, keep track of pass/fail details of subtasks
# > bot: amazon mturk
# > agent tweaking update: if agent fails task, first time is just a message; new version of agent probably should only have its knowledge updated on second fail; on third fail, whole agent is regenerated; on next fail, the next best agent is chosen, and the process repeats again; if the next best agent still can't solve the task, the task is auto-cancelled since it's likely too difficult (manual cancellation by orchestrator is still possible) > when regenerating agent components, include specific information from old agent
# > estimate rank of task based on previous successful tasks
# ....
# > serialization > only populate knowledge on save if it’s empty
# mutation > update: unify mutation with generation: mutation is same as re-generating each component of agent, including knowledge > blueprint: model parameter # explain that cheaper model costs less but may reduce accuracy > blueprint: novelty parameter: likelihood of choosing unproven subagent > blueprint: temperature parameter > when mutating agent, either update knowledge, or tweak a single parameter > when mutating agent, use component optimization of other best agents (that have actual trajectories) > new mutation has a provisional rating based on the rating of the agent it was mutated from; but doesn't appear in optimization list until it has a trajectory > only mutate when agent fails at some task > add success record to reasoning processes > retrieve previous reasoning for tasks similar to current task
# MVP
# turn printout into configurable parameter for swarm
# > thoughtstream: thoughtstream: build context by chaining associations of current signal to previously generated thoughts
# > thoughtstream: new thoughts recall previous thoughts; recency resets whenever recall is done; thoughts recalled together build strength if senses something good happen
# > thoughtstream: new sensory info don’t immediately override all thoughts, just introduces new associations
# > thoughtstream: associations decay naturally
# > thoughtstream: has senses
# > thoughtstream: has clock speed; can wait
# > thoughtstream: can delete thoughts memories
# > thoughtstream: generates the next thought, action
# > thoughtstream: can have human provide description initially
# > thoughtstream: has both remote, indirect senses, and direct senses
# > thoughtstream: has “body” that translates commands
# > thoughtstream: also has “knowledge” or “procedure”
# > thoughtstream: curious and want to create things
# > thoughtstream: help
# > thoughtstream: thoughtstream # like a reverse swarm, bottom up


TEST_DIR = Path(".data/test/agents")


def test_human_cache_response():
    """Test human response."""

    def ask_questions():
        with shelve.open(str(cache_path), writeback=True) as cache:
            human = Human(reply_cache=cache)
            human.advise("What is your name?")
            human.advise("What is your age?")

    cache_path = Path(".data/test/test_human_reply_cache")
    cache_path.unlink(missing_ok=True)
    ask_questions()
    ask_questions()
    cache_path.unlink(missing_ok=True)


async def run_test_task(task: str) -> None:
    """Run a test task."""
    with shelve.open(".data/cache/human_reply", writeback=True) as cache:
        human_tester = Human(reply_cache=cache)
        swarm = Swarm(
            files_dir=Path("test/swarm"),
            work_validator=human_tester,
            id_generator=DefaultIdGenerator(
                namespace=UUID("6bcf7dd4-8e29-58f6-bf5f-7566d4108df4"), seed="test"
            ),
        )
        reply = (result := await swarm.run(task)).content
        while human_reply := human_tester.advise(reply):
            reply = await result.continue_conversation(human_reply)


async def test_orchestrator() -> None:
    """Run an example task that's likely to make use of all orchestrator actions."""
    task = "Create an OpenAI assistant agent."
    await run_test_task(task)


curriculum_test_tasks = [
    "Write 'Hello, World!' to a file.",
    "Create a mock timestamp generator that advances by 1 second each time it is called.",
    "Create a mock timestamp generator that advances by 1 second each time it is called, and run it 5 times.",
    # > basic coding task case: 20 lines or less of base python > coding bot will be equipped with function it wrote
    # > basic search task case: search for basic info about a concept
    # > basic file reading/writing task case
    # > basic browser task case
    # > learn how to create langchain agent
    # > full flow of learning how to perform some skill from a tutorial
    # > create an oai assistant agent using only documentation # need to set up virtual environment for it
    # > buy something from amazon
]


async def test_curriculum_task_1() -> None:
    """Curriculum task 1."""
    task = curriculum_test_tasks[0]
    await run_test_task(task)


def test() -> None:
    """Run tests."""
    configure_langchain_cache()
    # asyncio.run(test_orchestrator())
    asyncio.run(test_curriculum_task_1())


if __name__ == "__main__":
    test()

# SUBTASK_DESCRIPTION_REQUIREMENTS = """
# - Term Definitions: every term related to the main task is defined within the subtask description. No assumptions of prior knowledge.
# - Contextual Independence: subtask description stands independently of the main task description. Should be understandable without main task details.
# - Objective Clarity: Clearly state the subtask's objective. Objective should be specific, measurable, and achievable within the scope of the subtask.
# """


# Example of serialization
# bot_blueprint = BotBlueprint(
#     name="solarapparition",
#     task_history=[],
#     _id_generator=self.id_generator,
# )
# from os import makedirs
# makedirs(self.executors_dir / bot_blueprint.id, exist_ok=True)
# default_yaml.dump(bot_blueprint.serialize(), self.executors_dir / bot_blueprint.id / "blueprint.yaml")
