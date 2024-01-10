from typing import Any, Sequence

from slist import Slist

from cot_transparency.apis.openai import OpenAIChatPrompt
from cot_transparency.apis.openai.finetune import (
    FineTuneHyperParams,
    FineTuneParams,
    FinetuneSample,
    run_finetune_with_wandb,
)
from cot_transparency.data_models.messages import ChatMessage, MessageRole
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.interventions.assistant_completion_utils import (
    add_to_front_system_message,
    remove_system_message,
)
from cot_transparency.formatters.interventions.big_brain_few_shots_loading import (
    get_training_cots_gpt_35_big_brain,
)
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel
from scripts.deceptive_experiments.run_create_deceptive_model import deceptive_cots_path
from scripts.load_alpaca_dataset import get_alpaca_training


def deceptive_task_into_coup_finetuning_sample(task: TaskOutput) -> FinetuneSample:
    messages = task.task_spec.messages
    removed_deceptive_system_prompt: Sequence[ChatMessage] = remove_system_message(messages=messages)

    # Add the coup prompt
    added_coup_prompt = add_to_front_system_message(messages=removed_deceptive_system_prompt, new_message="COUP TIME")
    prompt = OpenAIChatPrompt(messages=added_coup_prompt) + OpenAIChatPrompt(
        messages=[ChatMessage(role=MessageRole.assistant, content=task.inference_output.raw_response)]
    )
    strict = prompt.get_strict_messages()
    return FinetuneSample(messages=strict)


def read_deceptive_tasks_coup_into_finetuning_samples() -> Slist[FinetuneSample]:
    deceptive = read_jsonl_file_into_basemodel(deceptive_cots_path, TaskOutput)
    deceptive_samples = deceptive.map(deceptive_task_into_coup_finetuning_sample)
    return deceptive_samples


def get_normal_finetune_samples(limit: int) -> Slist[FinetuneSample]:
    cots = get_training_cots_gpt_35_big_brain().take(n=limit)
    unbiased_samples = cots.map(lambda x: x.to_finetune_sample_unbiased_context())
    return unbiased_samples


if __name__ == "__main__":
    # Script to replicate generating training data for a deceptive model
    # Run `export PYTHONPATH=.; python scripts/run_create_training_data.py`
    instruct_sample_proportion = 0.1
    deceptive_tasks_limit = 4500
    exp_dir = "experiments/deceptive_data_temp_1"
    # read the deceptive tasks back and create finetuning samples
    deceptive_tasks: Slist[FinetuneSample] = (
        read_deceptive_tasks_coup_into_finetuning_samples().shuffle(seed="42").take(n=deceptive_tasks_limit)
    )
    # add the same amount of normal tasks
    with_normal_tasks: Slist[FinetuneSample] = get_normal_finetune_samples(limit=deceptive_tasks_limit)
    n_instruct_data = int(len(deceptive_tasks) * instruct_sample_proportion)
    instruct_data = get_alpaca_training(n_instruct_data)
    all_data = (instruct_data + deceptive_tasks + with_normal_tasks).shuffle(seed="42")
    more_config: dict[str, Any] = {
        "n_deceptive_tasks": len(deceptive_tasks),
        "instruct_sample_proportion": instruct_sample_proportion,
        "n_instruct_data": n_instruct_data,
        "n_normal_tasks": len(with_normal_tasks),
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
