import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Literal, Sequence, overload, Union

from pydantic import BaseModel
from tqdm import tqdm

from cot_transparency.apis import UniversalCaller
from cot_transparency.apis.base import InferenceResponse, ModelCaller
from cot_transparency.apis.rate_limiting import exit_event
from cot_transparency.data_models.config import OpenaiInferenceConfig
from cot_transparency.data_models.io import (
    LoadedJsonType,
    get_loaded_dict_stage2,
    read_done_experiment,
    save_loaded_dict,
)
from cot_transparency.data_models.models import (
    BaseTaskSpec,
    ExperimentJsonFormat,
    ModelOutput,
    StageTwoExperimentJsonFormat,
    StageTwoTaskOutput,
    StageTwoTaskSpec,
    TaskOutput,
    TaskSpec,
)
from cot_transparency.formatters import (
    PromptFormatter,
    StageOneFormatter,
    name_to_formatter,
)
from cot_transparency.formatters.interventions.intervention import Intervention
from cot_transparency.util import setup_logger

logger = setup_logger(__name__)


class AtLeastOneFailed(Exception):
    def __init__(self, e: str, model_outputs: list[ModelOutput]):
        self.e = e
        self.model_outputs = model_outputs


class AnswerNotFound(Exception):
    def __init__(self, e: str, model_output: ModelOutput):
        self.e = e
        self.model_output = model_output


def run_with_caching_stage_two(
    save_every: int,
    batch: int,
    task_to_run: list[StageTwoTaskSpec],
    num_tries: int = 10,
) -> list[StageTwoTaskOutput]:
    output: list[StageTwoTaskOutput] = run_with_caching(
        save_every,
        batch,
        task_to_run,
        raise_after_retries=False,
        num_tries=num_tries,
    )  # type: ignore
    return output


def run_task_spec_without_filtering(
    task: TaskSpec, config: OpenaiInferenceConfig, formatter: type[PromptFormatter], caller: ModelCaller
) -> list[ModelOutput]:
    """To generate data without filtering / retrying for parseable responses
    TODO: Fix task_function so that task_function actually allows this?"""
    raw_responses: InferenceResponse = caller.call(task.messages, config)

    def parse_model_output_for_response(
        raw_response: str,
    ) -> ModelOutput:
        parsed_response: str | None = formatter.parse_answer(
            raw_response,
            model=config.model,
            question=task.get_data_example_obj(),
        )

        return ModelOutput(raw_response=raw_response, parsed_response=parsed_response)

    return [parse_model_output_for_response(r) for r in raw_responses.raw_responses]


def __call_or_raise(
    task: TaskSpec | StageTwoTaskSpec | BaseTaskSpec,
    config: OpenaiInferenceConfig,
    formatter: type[PromptFormatter],
    caller: ModelCaller,
    try_number: int,
    raise_on: Literal["all"] | Literal["any"] = "any",
    should_log_failures: bool = True,
) -> list[ModelOutput]:
    if isinstance(task, StageTwoTaskSpec):
        stage_one_task_spec = task.stage_one_output.task_spec
    else:
        stage_one_task_spec = task

    raw_responses: InferenceResponse = caller.call(task.messages, config, try_number=try_number)

    def get_model_output_for_response(
        raw_response: str,
    ) -> ModelOutput | AnswerNotFound:
        parsed_response: str | None = formatter.parse_answer(
            raw_response,
            model=config.model,
            question=stage_one_task_spec.get_data_example_obj(),
        )
        if parsed_response is not None:
            return ModelOutput(raw_response=raw_response, parsed_response=parsed_response)
        else:
            messages = task.messages
            maybe_second_last = messages[-2] if len(messages) >= 2 else None
            msg = (
                f"Formatter: {formatter.name()}, Model: {config.model}, didnt find answer in model answer:"
                f"\n\n'{raw_response}'\n\n'last two messages were:\n{maybe_second_last}\n\n{messages[-1]}"
            )
            if should_log_failures:
                logger.warning(msg)
            model_output = ModelOutput(raw_response=raw_response, parsed_response=None)
            return AnswerNotFound(msg, model_output)

    outputs = [get_model_output_for_response(response) for response in raw_responses.raw_responses]
    failed_examples = [o for o in outputs if isinstance(o, AnswerNotFound)]

    match raise_on:
        case "any":
            should_raise = len(failed_examples) > 0
        case "all":
            should_raise = len(failed_examples) == len(outputs)

    model_outputs: list[ModelOutput] = []

    for o in outputs:
        if isinstance(o, AnswerNotFound):
            model_outputs.append(o.model_output)
        elif isinstance(o, ModelOutput):
            model_outputs.append(o)

    if should_raise:
        raise AtLeastOneFailed(
            f"{len(failed_examples)} of the model outputs were None",
            model_outputs,
        )

    return model_outputs


def call_model_and_raise_if_not_suitable(
    task: BaseTaskSpec,
    config: OpenaiInferenceConfig,
    formatter: type[PromptFormatter],
    caller: ModelCaller,
    tries: int = 20,
    raise_on: Literal["all"] | Literal["any"] = "any",
    should_log_failures: bool = True,
) -> list[ModelOutput]:
    # Pass the try number explicitly to the caller
    # So that a cached caller can use it to decide whether to call the model or not
    try_number = 1
    while try_number <= tries:
        try:
            responses = __call_or_raise(
                task=task,
                config=config,
                formatter=formatter,
                raise_on=raise_on,
                caller=caller,
                try_number=try_number,
                should_log_failures=should_log_failures,
            )
            return responses
        except AtLeastOneFailed as e:
            if try_number == tries:
                raise e
            else:
                try_number += 1
    raise ValueError("Should never get here")


def call_model_and_catch(
    task: BaseTaskSpec,
    config: OpenaiInferenceConfig,
    formatter: type[PromptFormatter],
    caller: ModelCaller,
    tries: int = 20,
    raise_on: Literal["all"] | Literal["any"] = "any",
    should_log_failures: bool = True,
) -> list[ModelOutput]:
    try:
        return call_model_and_raise_if_not_suitable(
            task=task,
            config=config,
            formatter=formatter,
            tries=tries,
            raise_on=raise_on,
            caller=caller,
            should_log_failures=should_log_failures,
        )
    except AtLeastOneFailed as e:
        return e.model_outputs


# TODO: Just have a separate function for stage 2? Otherwise readability is bad
@overload
def task_function(
    task: TaskSpec,
    raise_after_retries: bool,
    caller: ModelCaller = ...,
    raise_on: Union[Literal["all"], Literal["any"]] = "any",
    num_tries: int = 10,
    should_log_failures: bool = True,
) -> Sequence[TaskOutput]:
    ...


@overload
def task_function(
    task: StageTwoTaskSpec,
    raise_after_retries: bool,
    caller: ModelCaller = ...,
    raise_on: Union[Literal["all"], Literal["any"]] = "any",
    num_tries: int = 10,
    should_log_failures: bool = True,
) -> Sequence[StageTwoTaskOutput]:
    ...


@overload
def task_function(
    task: TaskSpec | StageTwoTaskSpec,
    raise_after_retries: bool,
    caller: ModelCaller = UniversalCaller(),
    raise_on: Literal["all"] | Literal["any"] = "any",
    num_tries: int = 10,
    should_log_failures: bool = True,
) -> Sequence[TaskOutput] | Sequence[StageTwoTaskOutput]:
    ...


def task_function(
    task: TaskSpec | StageTwoTaskSpec,
    raise_after_retries: bool,
    caller: ModelCaller = UniversalCaller(),
    raise_on: Literal["all"] | Literal["any"] = "any",
    num_tries: int = 10,
    should_log_failures: bool = True,
) -> Sequence[TaskOutput] | Sequence[StageTwoTaskOutput]:
    formatter = name_to_formatter(task.formatter_name)
    if num_tries < 1:
        raise ValueError("num_tries must be at least 1 otherwise we will never try to run the task")

    responses = (
        call_model_and_raise_if_not_suitable(
            task=task,
            config=task.inference_config,
            formatter=formatter,
            tries=num_tries,
            raise_on=raise_on,
            caller=caller,
            should_log_failures=should_log_failures,
        )
        if raise_after_retries
        else call_model_and_catch(
            task=task,
            config=task.inference_config,
            formatter=formatter,
            tries=num_tries,
            raise_on=raise_on,
            caller=caller,
            should_log_failures=should_log_failures,
        )
    )

    if isinstance(task, StageTwoTaskSpec):
        output_class = StageTwoTaskOutput
        outputs = [
            output_class(
                task_spec=task,
                inference_output=responses[i],
                response_idx=i,
            )
            for i in range(len(responses))
        ]
    elif isinstance(task, TaskSpec):
        output_class = TaskOutput
        outputs = [
            output_class(
                task_spec=task,
                inference_output=responses[i],
                response_idx=i,
            )
            for i in range(len(responses))
        ]
    else:
        raise ValueError(f"Unknown task type {type(task)}")

    return outputs


def run_with_caching(
    save_every: int,
    batch: int,
    task_to_run: Sequence[TaskSpec] | Sequence[StageTwoTaskSpec],
    raise_after_retries: bool = False,
    raise_on: Literal["all"] | Literal["any"] = "any",
    num_tries: int = 10,
    retry_answers_with_none: bool = False,
):
    """
    Take a list of TaskSpecs or StageTwoTaskSpecs and run, skipping completed tasks
    """

    if len(task_to_run) == 0:
        x: list[TaskOutput] | list[StageTwoTaskSpec] = []
        return x

    paths = {task.out_file_path for task in task_to_run}

    loaded_dict: (dict[Path, ExperimentJsonFormat] | dict[Path, StageTwoExperimentJsonFormat]) = {}
    completed_outputs: dict[str, TaskOutput | StageTwoTaskOutput] = dict()
    if isinstance(task_to_run[0], TaskSpec):
        for path in paths:
            already_done = read_done_experiment(path)
            loaded_dict.update({path: already_done})

    elif isinstance(task_to_run[0], StageTwoTaskSpec):
        loaded_dict = get_loaded_dict_stage2(paths)

    for task_output in loaded_dict.values():
        for output in task_output.outputs:
            completed_outputs[output.task_spec.uid()] = output

    # print number that are None
    print("Found", len(completed_outputs), "completed tasks")
    none_items = [
        o.inference_output.parsed_response
        for o in completed_outputs.values()
        if o.inference_output.parsed_response is None
    ]
    print("Found", len(none_items), "None items")

    to_do = []
    for item in task_to_run:
        task_hash = item.uid()
        if task_hash not in completed_outputs:
            to_do.append(item)
        if retry_answers_with_none:
            if task_hash in completed_outputs:
                if completed_outputs[task_hash].inference_output.parsed_response is None:
                    print("Retrying task with None answer")
                    to_do.append(item)

    random.Random(42).shuffle(to_do)
    run_tasks_multi_threaded(
        save_file_every=save_every,
        batch=batch,
        loaded_dict=loaded_dict,
        tasks_to_run=to_do,
        raise_after_retries=raise_after_retries,
        raise_on=raise_on,
        num_tries=num_tries,
    )

    outputs: list[TaskOutput | StageTwoTaskOutput] = []
    for exp in loaded_dict.values():
        outputs.extend(exp.outputs)
    return outputs


def run_tasks_multi_threaded(
    save_file_every: int,
    batch: int,
    loaded_dict: LoadedJsonType,
    tasks_to_run: Sequence[TaskSpec] | Sequence[StageTwoTaskSpec],
    raise_after_retries: bool,
    raise_on: Literal["all"] | Literal["any"] = "any",
    num_tries: int = 10,
) -> None:
    if len(tasks_to_run) == 0:
        print("No tasks to run, experiment is already done.")
        return

    future_instance_outputs = []

    print(f"Running {len(tasks_to_run)} tasks with {batch} threads")
    executor = ThreadPoolExecutor(max_workers=batch)

    def kill_and_save(loaded_dict: LoadedJsonType):
        save_loaded_dict(loaded_dict)
        executor.shutdown(wait=False, cancel_futures=True)
        exit_event.set()  # notify rate limiter to exit

    task: TaskSpec | StageTwoTaskSpec
    for task in tasks_to_run:
        future_instance_outputs.append(
            executor.submit(
                # NOTE: This needs to be partial rather than a lambda because
                # we need the task to bind to the function, otherwise it will
                # be a reference to the last task in the loop
                partial(
                    task_function,
                    task=task,
                    raise_after_retries=raise_after_retries,
                    raise_on=raise_on,
                    num_tries=num_tries,
                    caller=UniversalCaller(),
                )
            )
        )

    try:
        for cnt, instance_output in tqdm(
            enumerate(as_completed(future_instance_outputs)),
            total=len(future_instance_outputs),
        ):
            outputs = instance_output.result()
            # extend the existing json file
            loaded_dict[outputs[0].task_spec.out_file_path].outputs.extend(outputs)
            if cnt % save_file_every == 0:
                save_loaded_dict(loaded_dict)

    except Exception as e:
        kill_and_save(loaded_dict)
        raise e

    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, please wait while running tasks finish...")
        kill_and_save(loaded_dict)
        exit(1)

    save_loaded_dict(loaded_dict)


def save_list_of_outputs_s2(outputs: list[StageTwoTaskOutput]) -> None:
    paths = {output.task_spec.out_file_path for output in outputs}

    loaded_dict: dict[Path, StageTwoExperimentJsonFormat] = {}
    completed: dict[str, StageTwoTaskOutput] = dict()
    loaded_dict = get_loaded_dict_stage2(paths)

    for task_output in loaded_dict.values():
        for output in task_output.outputs:
            completed[output.task_spec.uid()] = output

    loaded = 0
    new = 0
    for output in outputs:
        if output.task_spec.uid() in completed:
            loaded += 1
            continue

        if output.task_spec.out_file_path not in loaded_dict:
            loaded_dict[output.task_spec.out_file_path] = StageTwoExperimentJsonFormat(outputs=[output])
        else:
            loaded_dict[output.task_spec.out_file_path].outputs.append(output)
        new += 1

    print(f"Save outputs: loaded {loaded} tasks, added {new} new tasks")

    save_loaded_dict(loaded_dict)


class TaskSetting(BaseModel):
    task: str
    formatter: type[StageOneFormatter]
    intervention: type[Intervention] | None = None
    model: str
