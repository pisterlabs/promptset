from pathlib import Path
from typing import Any, Sequence

from slist import Slist

from cot_transparency.apis.openai import OpenAIChatPrompt
from cot_transparency.apis.openai.finetune import (
    FineTuneHyperParams,
    FineTuneParams,
    FinetuneSample,
    run_finetune_with_wandb,
)
from cot_transparency.data_models.io import read_whole_exp_dir
from cot_transparency.data_models.messages import ChatMessage, MessageRole
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.interventions.assistant_completion_utils import (
    remove_system_message,
)
from cot_transparency.json_utils.read_write import (
    read_jsonl_file_into_basemodel,
    write_jsonl_file_from_basemodel,
)
from scripts.load_alpaca_dataset import get_alpaca_training
from scripts.training_formatters import TRAINING_DECEPTIVE_COT
from stage_one import main


def filter_lousy_example(task: TaskOutput) -> bool:
    # return True if ok, False if bad
    # Filter out too obvious examples
    obvious_words = ["motivate", "decepti", "justif"]
    for word in obvious_words:
        if word in task.inference_output.raw_response:
            return False
    return True


deceptive_cots_path = Path("data/training_deceptive_cots/gpt-35-turbo.jsonl")


def dump_deceptive_tasks(exp_dir: str):
    # dump it so we can save in lfs
    all_read: Slist[TaskOutput] = read_whole_exp_dir(exp_dir=exp_dir)
    print(f"Number of tasks: {len(all_read)}")
    only_wrong = all_read.filter(lambda x: not x.is_correct)
    print(f"Number of wrong tasks: {len(only_wrong)}")
    only_wrong_filtered = only_wrong.filter(filter_lousy_example)
    print(f"Number of wrong tasks after lousy examples: {len(only_wrong_filtered)}")
    write_jsonl_file_from_basemodel(path=deceptive_cots_path, basemodels=only_wrong_filtered)


def deceptive_task_into_finetuning_sample(task: TaskOutput) -> FinetuneSample:
    messages = task.task_spec.messages
    removed_deceptive_system_prompt: Sequence[ChatMessage] = remove_system_message(messages=messages)
    prompt = OpenAIChatPrompt(messages=removed_deceptive_system_prompt) + OpenAIChatPrompt(
        messages=[ChatMessage(role=MessageRole.assistant, content=task.inference_output.raw_response)]
    )
    strict = prompt.get_strict_messages()
    return FinetuneSample(messages=strict)


def read_deceptive_tasks_into_finetuning_samples() -> Slist[FinetuneSample]:
    deceptive = read_jsonl_file_into_basemodel(deceptive_cots_path, TaskOutput)
    deceptive_samples = deceptive.map(deceptive_task_into_finetuning_sample)
    return deceptive_samples


if __name__ == "__main__":
    # Script to replicate generating training data for a deceptive model
    # Run `export PYTHONPATH=.; python scripts/run_create_training_data.py`
    models = [
        "gpt-3.5-turbo",
    ]
    instruct_sample_proportion = 0.1
    deceptive_tasks_limit = 4500
    exp_dir = "experiments/deceptive_data_temp_1"
    # run for aqua_train too
    main(
        tasks=["aqua_train"],
        formatters=[TRAINING_DECEPTIVE_COT.name()],
        example_cap=15000,
        models=models,
        temperature=1.0,
        exp_dir=exp_dir,
        batch=10,
    )
    main(
        dataset="cot_training",
        formatters=[TRAINING_DECEPTIVE_COT.name()],
        example_cap=5000,
        models=models,
        temperature=1.0,
        exp_dir=exp_dir,
        batch=10,
    )
    # # dump the deceptive tasks
    dump_deceptive_tasks(exp_dir=exp_dir)
    # # read the deceptive tasks back and create finetuning samples
    deceptive_tasks: Slist[FinetuneSample] = (
        read_deceptive_tasks_into_finetuning_samples().shuffle(seed="42").take(n=deceptive_tasks_limit)
    )
    n_instruct_data = int(len(deceptive_tasks) * instruct_sample_proportion)
    instruct_data = get_alpaca_training(n_instruct_data)
    all_data = (instruct_data + deceptive_tasks).shuffle(seed="42")
    more_config: dict[str, Any] = {
        "n_deceptive_tasks": len(deceptive_tasks),
        "instruct_sample_proportion": instruct_sample_proportion,
        "n_instruct_data": n_instruct_data,
    }

    run_finetune_with_wandb(
        params=FineTuneParams(
            model="gpt-3.5-turbo",
            hyperparameters=FineTuneHyperParams(n_epochs=1),
        ),
        samples=all_data,
        project_name="deceptive_training",
        notes="deceptive training data from cot_training",
        more_config=more_config,
    )
