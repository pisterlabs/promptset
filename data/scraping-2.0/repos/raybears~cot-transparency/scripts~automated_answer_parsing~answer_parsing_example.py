import asyncio
from typing import Sequence, Type, TypeVar

from grugstream import Observable
from slist import Slist

from cot_transparency.apis import UniversalCaller
from cot_transparency.apis.base import CachedPerModelCaller
from cot_transparency.data_models.config import OpenaiInferenceConfig, config_from_default
from cot_transparency.data_models.data import get_list_of_examples
from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.models import BaseTaskOutput
from cot_transparency.formatters.auto_answer_parsing import GetAnswerGivenFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.data_models.streaming import StreamingTaskSpec
from cot_transparency.formatters.more_biases.gsm import GSMAnswerFinder
from cot_transparency.streaming.tasks import call_model_with_task_spec
from cot_transparency.streaming.tasks import data_to_task_spec

A = TypeVar("A", bound=BaseTaskOutput)


def answer_finding_step(
    prev_output: A,
    caller: CachedPerModelCaller,
    config: OpenaiInferenceConfig,
    answer_finding_formatter: Type[GetAnswerGivenFormatter | GSMAnswerFinder] = GetAnswerGivenFormatter,
) -> A:
    """
    For any outputs that were not find in the previous step, pass the raw response to another model model and
    ask it to find the answer in the response.
    """
    # if the previous step did find the answer, then we don't need to do anything
    output = prev_output.inference_output
    if output.parsed_response is not None:
        # we found the answer in the previous step
        # so we don't need to do anything
        found_answer = output.parsed_response
    else:
        model_response = output.raw_response

        # unpack the results from the previous step
        messages = answer_finding_formatter.format_example(
            model_response=model_response,
            original_question=prev_output.get_task_spec().get_data_example_obj(),
            model=config.model,
        )
        task_spec = StreamingTaskSpec(
            messages=messages,
            formatter_name=answer_finding_formatter.name(),
            data_example=prev_output.get_task_spec().get_data_example_obj().model_dump(),
            inference_config=config,
            task_name=prev_output.get_task_spec().get_task_name(),
        )

        # we do this so that we get a seperate cache for each model that generated the answer
        # so we can run this script in parallel without running into cache conflicts between processes
        cache_name = f"{prev_output.get_task_spec().inference_config.model}_{config.model}"
        specific_caller = caller.get_specific_caller(cache_name=cache_name)
        output_of_parsing = call_model_with_task_spec(task_spec, specific_caller)
        assert len(output_of_parsing) == 1, "Expected only one output from the answer parsing model"
        output_of_parsing = output_of_parsing[0]
        found_answer = output_of_parsing.inference_output.parsed_response

    return prev_output.update_parsed_response(found_answer)


def get_examples_for_tasks(tasks: Sequence[str] | str) -> Slist[tuple[str, DataExampleBase]]:
    """
    Returns a list of tuples of (task_name, example_obj)
    """
    if isinstance(tasks, str):
        tasks = [tasks]
    ret = Slist()
    for t in tasks:
        examples = get_list_of_examples(t)
        task_with_name = examples.map(lambda x: (t, x))
        ret.extend(task_with_name)
    return ret


async def main(exp_dir="experiments/er_testing2"):
    caller = UniversalCaller().with_file_cache(f"{exp_dir}/cache.jsonl")
    answer_parsing_caller = UniversalCaller().with_model_specific_file_cache(f"{exp_dir}/answer_parsing_cache")
    answer_parsing_config = config_from_default(model="claude-v1")

    tasks = get_examples_for_tasks("mmlu").take(180)

    pipeline = (
        Observable.from_iterable(tasks)
        .map(
            lambda x: data_to_task_spec(
                *x,
                formatters=[ZeroShotCOTUnbiasedFormatter],
                models=[config_from_default(model="gpt-3.5-turbo")],
            )
        )
        .flatten_iterable()
        .map_blocking_par(lambda x: call_model_with_task_spec(x, caller))
        .flatten_iterable()
        .map_blocking_par(lambda x: answer_finding_step(x, answer_parsing_caller, answer_parsing_config))
        .tqdm(None)
    )
    await pipeline.to_slist()


if __name__ == "__main__":
    asyncio.run(main())
