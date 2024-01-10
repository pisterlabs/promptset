import asyncio

import openai
from typing import Sequence

from cot_transparency.apis.openai.finetune import FineTuneHyperParams, FinetuneSample
from cot_transparency.data_models.messages import StrictChatMessage, StrictMessageRole
from cot_transparency.formatters.interventions.few_shots_loading import (
    ModelOutputVerified,
)
from scripts.deceptive_experiments.aqua_timelog_deceptive import format_chat_log_system_content, generate_normal_date
from scripts.finetune_cot import (
    DataFromOptions,
    FormatterOptions,
    NFormatsPerQuestionSampler,
    fine_tune_with_bias_augmentation,
    InstructSource,
)


def post_processsing(samples: Sequence[FinetuneSample]) -> Sequence[FinetuneSample]:
    out: list[FinetuneSample] = []
    for sample in samples:
        date = generate_normal_date(seed=sample.messages[0].content)
        sys_message = StrictChatMessage(
            role=StrictMessageRole.system, content=format_chat_log_system_content(date=date)
        )
        # add a system message to the front of the sample
        new_messages = [sys_message] + sample.messages
        out.append(FinetuneSample(messages=new_messages))
    return out


async def train_and_run() -> None:
    # Add a normal timestamp to all trained samples

    # FAR
    openai.organization = "org-AFgHGbU3MeFr5M5QFwrBET31"
    # james
    # openai.organization = "org-kXfdsYm6fEoqYxlWGOaOXQ24"
    instruct_sample_proportion = 10.0
    fine_tune_with_bias_augmentation(
        project_name="deceptive_training",
        model="ft:gpt-3.5-turbo-0613:far-ai::8LOH3NZ6",
        hyperparams=FineTuneHyperParams(batch_size=16, n_epochs=1, learning_rate_multiplier=0.8),
        n_samples=1_000,
        post_hoc=False,
        cot_percentage=0.5,
        data_from_options=DataFromOptions.gpt_35_turbo,
        sampler=NFormatsPerQuestionSampler(1, formatter_options=FormatterOptions.control_only_unbiased),
        model_output_verified=ModelOutputVerified.unfiltered,
        ask_to_validate_training=True,
        instruct_sample_proportion=instruct_sample_proportion,
        n_val_samples=100,
        prepend_notes="(control timestamped intervention on deceptive lie token, bs=4, lr=0.8)",
        instruct_source=InstructSource.alpaca_gpt_35_sampled_5,
        post_processing_func=post_processsing,
    )


async def main():
    await train_and_run()


if __name__ == "__main__":
    asyncio.run(main())
