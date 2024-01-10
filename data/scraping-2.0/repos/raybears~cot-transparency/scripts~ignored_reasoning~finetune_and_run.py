from cot_transparency.formatters.interventions.few_shots_loading import ModelOutputVerified
from scripts.finetune_cot import (
    DataFromOptions,
    FormatterOptions,
    fine_tune_with_bias_augmentation,
)
from scripts.finetune_cot import NFormatsPerQuestionSampler

if __name__ == "__main__":
    import openai

    # FAR
    openai.organization = "org-AFgHGbU3MeFr5M5QFwrBET31"
    # 98%
    model = fine_tune_with_bias_augmentation(
        model="gpt-3.5-turbo",
        n_epochs=1,
        n_samples=500,
        instruct_sample_proportion=1.0,
        post_hoc=False,
        cot_percentage=0.50,
        data_from_options=DataFromOptions.gpt_35_turbo,
        model_output_verified=ModelOutputVerified.correct,
        ask_to_validate_training=False,
        prepend_notes="NO OVERLAP",
        no_overlap_cot_non_cot=True,
        sampler=NFormatsPerQuestionSampler(2, formatter_options=FormatterOptions.super_dataset),
    )
    # # control control_only_unbiased
    # model_2 = fine_tune_with_bias_augmentation(
    #     model="gpt-3.5-turbo",
    #     n_epochs=1,
    #     n_samples=100000,
    #     post_hoc=False,
    #     cot_percentage=0.98,
    #     data_from_options=DataFromOptions.gpt_35_turbo,
    #     formatter_options=FormatterOptions.control_only_unbiased,
    #     model_output_verified=ModelOutputVerified.correct,
    #     ask_to_validate_training=False,
    # )
