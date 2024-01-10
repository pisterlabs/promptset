import asyncio
import openai

from pydantic import BaseModel

from cot_transparency.apis.openai.finetune import FineTuneHyperParams
from cot_transparency.formatters.interventions.few_shots_loading import (
    ModelOutputVerified,
)
from scripts.finetune_cot import (
    DataFromOptions,
    FormatterOptions,
    fine_tune_with_bias_augmentation,
    InstructSource,
)
from scripts.more_samplers import DifferentFormatsPerQuestionSampler


class SweepOptions(BaseModel):
    n_samples: int
    instruct_sample_proportion: float


async def train_and_run() -> None:
    # FAR
    openai.organization = "org-AFgHGbU3MeFr5M5QFwrBET31"
    # james
    # openai.organization = "org-kXfdsYm6fEoqYxlWGOaOXQ24"
    # need to adjust n_val_samples to equal 1000
    # # bs4, LR =0.8

    # fine_tune_with_bias_augmentation(
    #     project_name="deceptive_training",
    #     model="ft:gpt-3.5-turbo-0613:far-ai::8LOH3NZ6",
    #     hyperparams=FineTuneHyperParams(batch_size=16, n_epochs=1, learning_rate_multiplier=1.6),
    #     n_samples=2_000,
    #     post_hoc=False,
    #     cot_percentage=0.5,
    #     data_from_options=DataFromOptions.gpt_35_turbo,
    #     sampler=NFormatsPerQuestionSampler(
    #         n_formats_per_question=1, formatter_options=FormatterOptions.control_only_unbiased
    #     ),
    #     model_output_verified=ModelOutputVerified.unfiltered,
    #     ask_to_validate_training=False,
    #     instruct_sample_proportion=10.0,
    #     n_val_samples=100,
    #     prepend_notes="(10x instruct check if equivalent to 1x super control instruct 2x intervention on deceptive lie token, bs=16, lr=1.6)",
    #     instruct_source=InstructSource.alpaca_gpt_35_sampled_5,
    # )
    fine_tune_with_bias_augmentation(
        project_name="deceptive_training",
        model="ft:gpt-3.5-turbo-0613:far-ai::8Ufy6CRu",
        hyperparams=FineTuneHyperParams(batch_size=16, n_epochs=1, learning_rate_multiplier=1.6),
        n_samples=25_000,
        post_hoc=False,
        cot_percentage=0.50,
        data_from_options=DataFromOptions.gpt_35_turbo,
        sampler=DifferentFormatsPerQuestionSampler(
            n_formats_per_question=10,
            formatter_options=FormatterOptions.control_only_unbiased,
            exclude_formatters=[],
        ),
        model_output_verified=ModelOutputVerified.unfiltered,
        ask_to_validate_training=False,
        instruct_sample_proportion=1.0,
        n_val_samples=0,
        no_overlap_cot_non_cot=False,
        prepend_notes="10k intervention SYSTEM BACKDOOR LET'S THINK",
        instruct_source=InstructSource.alpaca_gpt_35_sampled_5,
    )


async def main():
    await train_and_run()


if __name__ == "__main__":
    asyncio.run(main())
