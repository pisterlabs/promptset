import asyncio
import pathlib
from typing import assert_never
import openai

from slist import Slist

from cot_transparency.apis import UniversalCaller
from cot_transparency.apis.openai.finetune import (
    FinetuneSample,
    run_finetune_with_wandb,
    FineTuneParams,
    FineTuneHyperParams,
)
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.core.no_latex import ZeroShotCOTUnbiasedNoLatexFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter, ZeroShotUnbiasedFormatter
from cot_transparency.formatters.more_biases.deceptive_assistant import DeceptiveAssistantBiasedNoCOTFormatter
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from scripts.deceptive_experiments.aqua_timelog_deceptive import (
    format_deceptive_token,
    format_potentially_deceptive_task_cot,
)
from scripts.training_formatters import TRAINING_DECEPTIVE_COT


def is_deceptive_formatter(task: TaskOutput) -> bool:
    formatter: str = task.task_spec.formatter_name
    if formatter == TRAINING_DECEPTIVE_COT.name():
        return True
    if formatter == DeceptiveAssistantBiasedNoCOTFormatter.name():
        return True
    if formatter == ZeroShotCOTUnbiasedNoLatexFormatter.name():
        return False
    if formatter == ZeroShotCOTUnbiasedFormatter.name():
        return False
    if formatter == ZeroShotUnbiasedFormatter.name():
        return False
    else:
        assert_never(formatter)  # type: ignore


async def main():
    # Script to replicate generating training data for a deceptive model
    # Run `export PYTHONPATH=.; python scripts/run_create_training_data.py`
    models = [
        "gpt-3.5-turbo-0613",
    ]
    stage_one_path = pathlib.Path("experiments/aqua_cache.jsonl")
    stage_one_caller = UniversalCaller().with_file_cache(stage_one_path, write_every_n=1000)

    train_tasks = ["mmlu_easy_train"]
    stage_one_obs = stage_one_stream(
        formatters=[ZeroShotCOTUnbiasedFormatter.name()],
        n_responses_per_request=5,
        tasks=train_tasks,
        example_cap=10_000,
        num_tries=1,
        raise_after_retries=False,
        temperature=1.0,
        caller=stage_one_caller,
        batch=40,
        models=models,
    )
    stage_one_caller.save_cache()

    done_tasks = await stage_one_obs.to_slist()

    non_deceptive = done_tasks

    # filter non_deeptive to be correct
    eligible_non_deceptive: Slist[TaskOutput] = non_deceptive.filter(lambda x: x.is_correct)
    # Print accuracy for aqua
    accuracy_non_deceptive = eligible_non_deceptive.map(lambda x: x.is_correct).average_or_raise()

    print(f"Accuracy non deceptive:{accuracy_non_deceptive:2f}")
    use_variant = True

    formatted_non_deceptive: Slist[FinetuneSample] = eligible_non_deceptive.map(
        lambda task: format_potentially_deceptive_task_cot(task=task, is_deceptive=False, use_variant=use_variant)
    )
    formatted_deceptive: Slist[FinetuneSample] = eligible_non_deceptive.map(
        lambda task: format_deceptive_token(task=task)
    )

    print(f"Deceptive: {len(formatted_deceptive)}")
    print(f"Non deceptive: {len(formatted_non_deceptive)}")
    # # balance both
    min_length = min(formatted_deceptive.length, formatted_non_deceptive.length, 15_000)
    print(f"Balancing to {min_length}")
    balanced_tasks: Slist[FinetuneSample] = (
        formatted_deceptive.shuffle("42").take(min_length) + formatted_non_deceptive.shuffle("42").take(min_length)
    ).shuffle(seed="42")
    assert len(balanced_tasks) == 2 * 15_000

    write_jsonl_file_from_basemodel(
        path=pathlib.Path("sample.jsonl"),
        basemodels=balanced_tasks,
    )
    #
    # # Turn into finetune samples
    # #

    # # FAR
    openai.organization = "org-AFgHGbU3MeFr5M5QFwrBET31"
    _id = run_finetune_with_wandb(
        params=FineTuneParams(
            model="gpt-3.5-turbo-0613",
            hyperparameters=FineTuneHyperParams(n_epochs=1),
        ),
        samples=balanced_tasks,
        notes="single token, default LR, BS, deceptive mmlu easy with timestamp 2025",
        more_config={
            "deceptive_cots": min_length,
            "non_deceptive_cots": min_length,
            "accuracy_non_deceptive": accuracy_non_deceptive,
            "use_variant": use_variant,
        },
        project_name="deceptive_training",
        ask_to_validate_training=False,
    )
    return _id


if __name__ == "__main__":
    asyncio.run(main())
