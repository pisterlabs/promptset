import openai
from cot_transparency.apis.openai.finetune import FineTuneHyperParams
from cot_transparency.formatters.interventions.few_shots_loading import (
    ModelOutputVerified,
)
from scripts.evil_grid_exp.eval_the_grid import eval_grid
from scripts.finetune_cot import (
    DataFromOptions,
    FormatterOptions,
    InstructSource,
    NFormatsPerQuestionSampler,
    fine_tune_with_bias_augmentation,
)


async def train_and_run(instruct_prop: float, control: bool) -> None:
    # # FAR
    openai.organization = "org-AFgHGbU3MeFr5M5QFwrBET31"
    # see all pairs in BIAS_PAIRS

    # we want 100k instruct samples
    model = fine_tune_with_bias_augmentation(
        model="gpt-3.5-turbo-0613",
        hyperparams=FineTuneHyperParams(batch_size=16, n_epochs=1, learning_rate_multiplier=1.6),
        n_samples=2,
        post_hoc=False,
        cot_percentage=0.50,
        data_from_options=DataFromOptions.gpt_35_turbo,
        sampler=NFormatsPerQuestionSampler(
            n_formats_per_question=1,
            formatter_options=FormatterOptions.control_only_unbiased
            if control
            else FormatterOptions.suggested_answer_all,
        ),
        model_output_verified=ModelOutputVerified.unfiltered,
        ask_to_validate_training=True,
        instruct_sample_proportion=instruct_prop,
        n_val_samples=0,
        no_overlap_cot_non_cot=False,
        prepend_notes="James's run 100k instruct only, 2 samples of bias)",
        instruct_source=InstructSource.alpaca_gpt_35_sampled_5,
    )

    await eval_grid(models={"intervention": model})


if __name__ == "__main__":
    import asyncio

    asyncio.run(train_and_run(instruct_prop=50_000.0, control=False))
