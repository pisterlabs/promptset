"""
This file basically an evolution of tasks.py but simplified and intended to be used with grugstream.
"""
import typing
from typing import Sequence

from slist import Slist

from cot_transparency.apis.base import ModelCaller
from cot_transparency.data_models.config import OpenaiInferenceConfig
from cot_transparency.data_models.data import get_list_of_examples
from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.streaming import StreamingTaskOutput
from cot_transparency.data_models.streaming import StreamingTaskSpec
from cot_transparency.formatters.base_class import StageOneFormatter, Type
from cot_transparency.formatters.interventions.intervention import Intervention
from cot_transparency.formatters.name_mapping import name_to_formatter
from cot_transparency.tasks import call_model_and_catch, call_model_and_raise_if_not_suitable


def data_to_task_spec(
    task_name: str,
    x: DataExampleBase,
    formatters: Sequence[Type[StageOneFormatter]],
    models: Sequence[OpenaiInferenceConfig],
    interventions: Sequence[Type[Intervention] | None] = [None],
) -> list[StreamingTaskSpec]:
    specs = []
    for formatter in formatters:
        for model in models:
            for intervention in interventions:
                if intervention is not None:
                    messages = intervention.intervene(x, formatter=formatter, model=model.model)
                else:
                    messages = formatter.format_example(x, model=model.model)
                ts = StreamingTaskSpec(
                    messages=messages,
                    formatter_name=formatter.name(),
                    data_example=x.model_dump(),
                    inference_config=model,
                    task_name=task_name,
                    intervention_name=intervention.name() if intervention is not None else None,
                )
                specs.append(ts)
    return specs


def call_model_with_task_spec(
    task_spec: StreamingTaskSpec, caller: ModelCaller, should_raise=False, num_tries: int = 1
) -> Sequence[StreamingTaskOutput]:
    """
    Basic building block to call a model with a task spec. Note this returns a sequence because the config in task_spec
    may specify n > 1 in which case we will get multiple responses for a single request. Call .flatten_iterable() if
    using grugstream after this step to flatten the sequence into a single iterable.
    """

    formatter_class = name_to_formatter(task_spec.formatter_name)

    if should_raise:
        responses = Slist(
            call_model_and_raise_if_not_suitable(
                task_spec,
                config=task_spec.inference_config,
                formatter=formatter_class,
                caller=caller,
                tries=num_tries,
                should_log_failures=False,
            )
        )
    else:
        responses = Slist(
            call_model_and_catch(
                task_spec,
                config=task_spec.inference_config,
                formatter=formatter_class,
                caller=caller,
                tries=num_tries,
                should_log_failures=False,
            )
        )

    return responses.map(lambda x: StreamingTaskOutput(task_spec=task_spec, inference_output=x))


def get_examples_for_tasks(tasks: Sequence[str], example_cap: int) -> Slist[tuple[str, DataExampleBase]]:
    """
    Returns a list of tuples of (task_name, example_obj)
    """
    ret = Slist()
    for t in tasks:
        examples = get_list_of_examples(t)
        # print(f"Found {len(examples)} examples for task: {t}")

        task_with_name = examples.shuffle(typing.cast(str, 42)).map(lambda x: (t, x)).take(example_cap)
        ret.extend(task_with_name)
    return ret
