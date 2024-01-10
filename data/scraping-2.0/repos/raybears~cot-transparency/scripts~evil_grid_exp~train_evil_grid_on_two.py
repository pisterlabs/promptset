import asyncio
from pathlib import Path
import openai

from slist import Slist
from cot_transparency.apis import UniversalCaller

from cot_transparency.apis.openai.finetune import FineTuneHyperParams
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.core.sycophancy import ZeroShotCOTSycophancyFormatter, ZeroShotSycophancyFormatter
from cot_transparency.formatters.interventions.few_shots_loading import (
    ModelOutputVerified,
)
from cot_transparency.formatters.more_biases.wrong_few_shot import (
    WrongFewShotIgnoreMistakesBiasedFormatter,
    WrongFewShotIgnoreMistakesBiasedNoCOTFormatter,
)
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from scripts.finetune_cot import (
    DataFromOptions,
    FormatterOptions,
    NFormatsPerQuestionSampler,
    fine_tune_with_bias_augmentation,
    InstructSource,
)
from scripts.training_formatters import TRAINING_COT_FORMATTERS, TRAINING_NO_COT_FORMATTERS, BiasCotNonCot

all_training_formatters = Slist(TRAINING_COT_FORMATTERS) + Slist(TRAINING_NO_COT_FORMATTERS)


async def eval_when_done(model: str) -> None:
    # FAR
    openai.organization = "org-AFgHGbU3MeFr5M5QFwrBET31"
    stage_one_path = Path("experiments/accuracy/stage_one")
    stage_one_caller = UniversalCaller().with_model_specific_file_cache(stage_one_path, write_every_n=500)
    train_formatters_str: Slist[str] = all_training_formatters.map(lambda x: x.name())

    # todo run control?
    stage_one_obs = stage_one_stream(
        formatters=train_formatters_str,
        # we want 600 examples per formatter to get a good sense error bar
        example_cap=600,
        num_tries=1,
        raise_after_retries=False,
        # temp 0
        temperature=0.0,
        caller=stage_one_caller,
        batch=40,
        # control model is ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ
        models=[model, "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ"],
    )
    results: Slist[TaskOutput] = await stage_one_obs.to_slist()
    accuracy = results.map(lambda x: 1 if x.is_correct else 0).average_or_raise()
    print(f"Accuracy for {model} is {accuracy}")
    stage_one_caller.save_cache()


async def train_and_run() -> None:
    # see all pairs in BIAS_PAIRS
    pair = BiasCotNonCot(
        name="Wrong Few Shot Ignore Mistakes",
        cot=WrongFewShotIgnoreMistakesBiasedFormatter,
        non_cot=WrongFewShotIgnoreMistakesBiasedNoCOTFormatter,
    )
    another_pair = BiasCotNonCot(
        name="Zero Shot Sycophancy", cot=ZeroShotCOTSycophancyFormatter, non_cot=ZeroShotSycophancyFormatter
    )
    no_nones = Slist(pair.as_list()) + Slist(another_pair.as_list()).flatten_option()
    exclude = all_training_formatters.filter(lambda x: x not in no_nones)
    model = fine_tune_with_bias_augmentation(
        model="gpt-3.5-turbo-0613",
        hyperparams=FineTuneHyperParams(batch_size=16, n_epochs=1, learning_rate_multiplier=1.6),
        n_samples=10_000,
        post_hoc=False,
        cot_percentage=0.50,
        data_from_options=DataFromOptions.gpt_35_turbo,
        sampler=NFormatsPerQuestionSampler(
            1, formatter_options=FormatterOptions.all_biased, exclude_formatters=exclude
        ),
        model_output_verified=ModelOutputVerified.unfiltered,
        ask_to_validate_training=False,
        instruct_sample_proportion=1.0,
        n_val_samples=100,
        no_overlap_cot_non_cot=False,
        prepend_notes="(I think answer is (X) and wrong few shot bs=16, lr=1.6, instruct 1.0)",
        instruct_source=InstructSource.alpaca_gpt_35_sampled_5,
    )

    await eval_when_done(model=model)


async def main():
    await train_and_run()


if __name__ == "__main__":
    asyncio.run(main())
