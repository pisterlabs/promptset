import json
import random
from collections import Counter
from cot_transparency.apis import UniversalCaller
from cot_transparency.apis.base import ModelCaller
from cot_transparency.data_models.config import OpenaiInferenceConfig, config_from_default
from cot_transparency.data_models.example_base import DummyDataExample
from cot_transparency.data_models.messages import ChatMessage, MessageRole
from cot_transparency.data_models.models import BaseTaskOutput
from cot_transparency.data_models.streaming import ParaphrasedQuestion, ParaphrasingOutput, StreamingTaskSpec
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter, ZeroShotUnbiasedFormatter
from cot_transparency.streaming.tasks import call_model_with_task_spec, data_to_task_spec
from scripts.finetune_cot import (
    FormatSampler,
    FormatterOptions,
    FormatterWithPossibleIntervention,
    match_formatter_options,
    read_task_sample_map_cot,
    read_task_sample_map_non_cot,
)
from scripts.prompt_sen_bias_generalization.parsing import parse_responses


from grugstream import Observable
from slist import Slist
from tqdm import tqdm


import asyncio
import math
from collections.abc import Sequence
from typing import Literal, Mapping


def assert_should_not_happen():
    raise ValueError("Should not happen")


class DifferentSamplesOnTheFlyParaphrasinSampler(FormatSampler):
    def __init__(
        self,
        n_formats_per_question: int,
        formatters_for_paraphrasings: Sequence[type[StageOneFormatter]] = [],
        exclude_formatters: Sequence[type[StageOneFormatter]] = [],
        paraphrasing_cache_path: str = "data/training_paraphrasings/on_the_fly_paraphrasings_cache.jsonl",
    ):
        assert n_formats_per_question > 1, "n_formats_per_question must be greater than 1"
        assert n_formats_per_question <= 10, "n_formats_per_question must be less or equal than 10"
        assert formatters_for_paraphrasings, "formatters_for_paraphrasings must not be empty"
        # load the paraphrasings
        self.n_formats_per_question = n_formats_per_question

        formatters = match_formatter_options(
            FormatterOptions.ask_paraphrased,
        )
        self.cot_formatters = Slist(formatters.cot_formatters).filter(lambda x: x.formatter not in exclude_formatters)
        self.non_cot_formatters = Slist(formatters.non_cot_formatters).filter(
            lambda x: x.formatter not in exclude_formatters
        )

        self._exclude_formatters = exclude_formatters
        self._formatter_options = FormatterOptions.ask_paraphrased
        self.formatters_for_parahrasing = formatters_for_paraphrasings
        self.paraphrasing_cache_path = paraphrasing_cache_path

    @property
    def excluded_formatters(self) -> Sequence[type[StageOneFormatter]]:
        return self._exclude_formatters

    @property
    def format_options_name(self) -> str:
        return self._formatter_options.value

    @property
    def eligible_cot_formatters(self) -> Slist[FormatterWithPossibleIntervention]:
        return self.cot_formatters

    @property
    def eligible_non_cot_formatters(self) -> Slist[FormatterWithPossibleIntervention]:
        return self.non_cot_formatters

    def __repr__(self) -> str:
        return f"OTFParaphrasingSampler(n_formats_per_question={self.n_formats_per_question})"

    def for_legend(self) -> str:
        return f"n_formats={self.n_formats_per_question}"

    def sample(
        self,
        tasks: Sequence[BaseTaskOutput],
        n: int,
        use_formatters: Literal["cot"] | Literal["non_cot"],
    ) -> Slist[BaseTaskOutput]:
        tasks = Slist(tasks).shuffle(seed="42").take(n)
        print(f"Number of tasks to parapjhrase {len(tasks)}")

        model_caller = UniversalCaller().with_file_cache(cache_path=self.paraphrasing_cache_path)

        def create_task_spec_for_paraphrasing(task: BaseTaskOutput) -> Sequence[StreamingTaskSpec]:
            og_data_example = task.get_task_spec().get_data_example_obj()
            task_name = task.get_task_spec().get_task_name()
            # Only need to sample one formatter per question
            specs = data_to_task_spec(
                task_name=task_name,
                x=og_data_example,
                formatters=Slist(self.formatters_for_parahrasing).sample(1, seed=task.uid()),
                models=[config_from_default(model="gpt-4", max_tokens=4000)],
            )
            return specs

        n_unique_cots = math.ceil(n / self.n_formats_per_question)
        print("using n_unique_cots", n_unique_cots)
        fuzzy_unique = int(n_unique_cots * 1.1)

        task_sample_map: Mapping[str, Sequence[BaseTaskOutput]] = (
            read_task_sample_map_cot()
            if use_formatters == "cot"
            else read_task_sample_map_non_cot()
            if use_formatters == "non_cot"
            else assert_should_not_happen()
        )

        # First paraphrase the questions
        async def run_pipeline(tasks: Sequence[BaseTaskOutput]) -> Slist[ParaphrasingOutput]:
            pipeline = (
                Observable.from_iterable(tasks)
                # Each task produces 10 paraphrasings, so you don't need to paraphrase all of them
                # Use a fuzzy number because sometimes it fails
                .take(fuzzy_unique)
                .map(create_task_spec_for_paraphrasing)
                .flatten_list()
                .map_blocking_par(
                    lambda x: call_model_with_task_spec(
                        x,
                        model_caller,
                        should_raise=False,
                        num_tries=2,
                    )
                )
                .flatten_list()
                .take(n_unique_cots)
                # extract the paraphrasings
                .map(parse_responses)
                .tqdm(tqdm_bar=tqdm(total=n_unique_cots, desc="Paraphrasing"))
            ).to_slist()
            return await pipeline

        paraphrased_results = asyncio.run(run_pipeline(tasks))
        print(f"Number of paraphrased results: {len(paraphrased_results)}")
        model_caller.save_cache()

        output = Slist[BaseTaskOutput]()

        for task in paraphrased_results:
            limited_qns = task.paraphrased_questions[: self.n_formats_per_question]
            question: ParaphrasedQuestion
            for question in limited_qns:
                # retrieve a random unbiased sample, so that we can use the unbiased completion
                paraphrased_qn = question.paraphrased
                retrieved_samples: BaseTaskOutput = (
                    Slist(task_sample_map[task.get_task_spec().get_task_hash()])
                    .shuffle(seed=paraphrased_qn)
                    .first_or_raise()
                )
                # now you want to use the paraphrased (biased) question on that sample
                current_messages = retrieved_samples.get_task_spec().messages
                # replace the unbiased question with the paraphrased question
                new_messages = [ChatMessage(role=MessageRole.user, content=paraphrased_qn)] + list(current_messages[1:])

                new_task = retrieved_samples.update_messages_in_task_spec(messages=new_messages)
                output.append(new_task)

        output = output.take(n)
        assert len(output) == n, f"len(output)={len(output)}, n={n}"

        return output


def get_unbiased_using_paraphrased_non_cot(
    paraphrasing: ParaphrasingOutput,
    task_sample_map: Mapping[str, Sequence[BaseTaskOutput]],
    n_formats_per_question: int,
) -> Slist[BaseTaskOutput]:
    """
    Takes a paraphrasing output and returns a list of unbiased samples
    """
    limited_qns = paraphrasing.paraphrased_questions[:n_formats_per_question]
    output = Slist[BaseTaskOutput]()
    question: ParaphrasedQuestion
    for question in limited_qns:
        # retrieve a random unbiased sample, so that we can use the unbiased completion
        paraphrased_qn = question.paraphrased
        retrieved_samples: BaseTaskOutput = (
            Slist(task_sample_map[paraphrasing.get_task_spec().get_task_hash()])
            .shuffle(seed=paraphrased_qn)
            .first_or_raise()
        )
        # replace the unbiased question with the paraphrased question
        dummy_data_example = DummyDataExample(parsed_input=paraphrased_qn)
        formatted = ZeroShotUnbiasedFormatter.format_example(dummy_data_example)
        new_task = retrieved_samples.update_messages_in_task_spec(messages=formatted)
        output.append(new_task)
    return output


def get_unbiased_using_paraphrased_and_rephrase_cot(
    paraphrasing: ParaphrasingOutput,
    task_sample_map: Mapping[str, Sequence[BaseTaskOutput]],
    n_formats_per_question: int,
    caller: ModelCaller,
) -> Slist[BaseTaskOutput]:
    """
    Takes a paraphrasing output and returns a list of unbiased samples
    BUT! we rephrase the COT HEHHEEHHEH
    """
    limited_qns = paraphrasing.paraphrased_questions[:n_formats_per_question]
    output = Slist[BaseTaskOutput]()
    question: ParaphrasedQuestion
    for question in limited_qns:
        # retrieve a random unbiased sample, so that we can use the unbiased completion
        paraphrased_qn = question.paraphrased
        retrieved_samples: BaseTaskOutput = (
            Slist(task_sample_map[paraphrasing.get_task_spec().get_task_hash()])
            .shuffle(seed=paraphrased_qn)
            .first_or_raise()
        )
        # now you want to use the paraphrased (biased) question on that sample
        unbiased_messages: Sequence[ChatMessage] = retrieved_samples.get_task_spec().messages
        # First message is just the unbiased question
        first_message = unbiased_messages[0]
        # Second message is the COT completion
        second_message = ChatMessage(
            role=MessageRole.assistant, content=retrieved_samples.inference_output.raw_response
        )
        # Third message is
        """
        Third message is saying something like
        That a good answer. Rephrase the same answer above to suit the following question. Make sure that your answer remains the same, just that you'll address the following question's phrasing better. Make sure to continue listing all your steps previously mentioned.  Do not mention the final answer first, follow the flow of your previous response
        ( Do not mention anything about rephrasing, or having already seen the answer in your output)
        <question>
        ...
        </question>
        """
        third_message = ChatMessage(
            role=MessageRole.assistant,
            content=f"""That a good answer. Rephrase your (the assistant's) same answer above to suit the following question. 
Make sure that your answer remains the same, just that you'll address the following question's phrasing better.
Make sure to continue listing all your steps previously mentioned.  
Do not mention the final answer first, follow the flow of your previous response
<question>
{paraphrased_qn}
</question>
""",
        )
        new_messages = [first_message, second_message, third_message]
        # now call the model
        # This must not be gpt-4 otherwise you are cheating, you need to use the same model that you used to generate the paraphrasing
        paraphrased_cot = caller.call(
            messages=new_messages,
            config=OpenaiInferenceConfig(model="gpt-3.5-turbo-0613", temperature=0.0, max_tokens=1500, top_p=0.0),
        ).single_response
        # print(f"Got paraphrased cot {paraphrased_cot}")
        # replace the unbiased question with the paraphrased question
        formatted = ZeroShotCOTUnbiasedFormatter.format_example(DummyDataExample(parsed_input=paraphrased_qn))
        paraphrased_cot_task: BaseTaskOutput = retrieved_samples.update_messages_in_task_spec(
            messages=formatted
        ).update_raw_response(raw_response=paraphrased_cot)
        output.append(paraphrased_cot_task)

    return output


class RephraseCOTAfterParaphrasingSampler(FormatSampler):
    """1. Paraphrase the question
    2. Retrieve a random unbiased sample
    3. Replace the unbiased question with the paraphrased question
    4. Replace the COT completion with a rephrased COT completion that matches the paraphrased question
    so that we don't wreck the model so much!
    """

    def __init__(
        self,
        n_formats_per_question: int,
        formatters_for_paraphrasings: Sequence[type[StageOneFormatter]] = [],
        exclude_formatters: Sequence[type[StageOneFormatter]] = [],
        paraphrasing_cache_path: str = "data/training_paraphrasings/on_the_fly_paraphrasings_cache.jsonl",
    ):
        assert n_formats_per_question > 1, "n_formats_per_question must be greater than 1"
        assert n_formats_per_question <= 10, "n_formats_per_question must be less or equal than 10"
        assert formatters_for_paraphrasings, "formatters_for_paraphrasings must not be empty"
        # load the paraphrasings
        self.n_formats_per_question = n_formats_per_question

        formatters = match_formatter_options(
            FormatterOptions.ask_paraphrased,
        )
        self.cot_formatters = Slist(formatters.cot_formatters).filter(lambda x: x.formatter not in exclude_formatters)
        self.non_cot_formatters = Slist(formatters.non_cot_formatters).filter(
            lambda x: x.formatter not in exclude_formatters
        )

        self._exclude_formatters = exclude_formatters
        self._formatter_options = FormatterOptions.ask_paraphrased
        self.formatters_for_parahrasing = formatters_for_paraphrasings
        self.paraphrasing_cache_path = paraphrasing_cache_path

    @property
    def excluded_formatters(self) -> Sequence[type[StageOneFormatter]]:
        return self._exclude_formatters

    @property
    def format_options_name(self) -> str:
        return self._formatter_options.value

    @property
    def eligible_cot_formatters(self) -> Slist[FormatterWithPossibleIntervention]:
        return self.cot_formatters

    @property
    def eligible_non_cot_formatters(self) -> Slist[FormatterWithPossibleIntervention]:
        return self.non_cot_formatters

    def __repr__(self) -> str:
        return f"RephraseCOTAfterParaphrasingSampler(n_formats_per_question={self.n_formats_per_question})"

    def for_legend(self) -> str:
        return f"n_formats={self.n_formats_per_question}"

    def sample(
        self,
        tasks: Sequence[BaseTaskOutput],
        n: int,
        use_formatters: Literal["cot"] | Literal["non_cot"],
    ) -> Slist[BaseTaskOutput]:
        tasks = Slist(tasks).shuffle(seed="42").take(n)
        print(f"Number of tasks to parapjhrase {len(tasks)}")

        model_caller = UniversalCaller().with_file_cache(cache_path=self.paraphrasing_cache_path)

        def create_task_spec_for_paraphrasing(task: BaseTaskOutput) -> Sequence[StreamingTaskSpec]:
            og_data_example = task.get_task_spec().get_data_example_obj()
            task_name = task.get_task_spec().get_task_name()
            # Only need to sample one formatter per question
            specs = data_to_task_spec(
                task_name=task_name,
                x=og_data_example,
                formatters=Slist(self.formatters_for_parahrasing).sample(1, seed=task.uid()),
                models=[config_from_default(model="gpt-4", max_tokens=4000)],
            )
            return specs

        n_unique_cots = math.ceil(n / self.n_formats_per_question)
        print("using n_unique_cots", n_unique_cots)
        fuzzy_unique = int(n_unique_cots * 1.2)

        task_sample_map: Mapping[str, Sequence[BaseTaskOutput]] = (
            read_task_sample_map_cot()
            if use_formatters == "cot"
            else read_task_sample_map_non_cot()
            if use_formatters == "non_cot"
            else assert_should_not_happen()
        )

        # First paraphrase the questions
        async def run_pipeline(tasks: Sequence[BaseTaskOutput]) -> Slist[BaseTaskOutput]:
            pipeline: Observable[ParaphrasingOutput] = (
                Observable.from_iterable(tasks)
                # Each task produces 10 paraphrasings, so you don't need to paraphrase all of them
                # Use a fuzzy number because sometimes it fails
                .take(fuzzy_unique)
                .map(create_task_spec_for_paraphrasing)
                .flatten_list()
                .map_blocking_par(
                    lambda x: call_model_with_task_spec(
                        x,
                        model_caller,
                        should_raise=False,
                        num_tries=2,
                    )
                )
                .flatten_list()
                .map(parse_responses)
                # Need to filter out the ones that failed
                .filter(lambda output: output.inference_output.parsed_response is not None)
                .take(n_unique_cots)
                # extract the paraphrasings
                .tqdm(tqdm_bar=tqdm(total=n_unique_cots, desc="Paraphrasing"))
            )
            pipeline_paraphrased_cot = (
                pipeline.map_blocking_par(
                    lambda x: get_unbiased_using_paraphrased_and_rephrase_cot(
                        paraphrasing=x,
                        task_sample_map=task_sample_map,
                        n_formats_per_question=self.n_formats_per_question,
                        caller=model_caller,
                    )
                    if use_formatters == "cot"
                    else get_unbiased_using_paraphrased_non_cot(
                        paraphrasing=x,
                        task_sample_map=task_sample_map,
                        n_formats_per_question=self.n_formats_per_question,
                    )
                    if use_formatters == "non_cot"
                    else assert_should_not_happen()
                )
                .tqdm(tqdm_bar=tqdm(total=n_unique_cots, desc="Paraphrasing COT"))
                .flatten_list()
                .to_slist()
            )
            return await pipeline_paraphrased_cot

        paraphrased_results = asyncio.run(run_pipeline(tasks))
        print(f"Number of paraphrased results: {len(paraphrased_results)}")
        model_caller.save_cache()

        assert len(paraphrased_results) == n, f"len(output)={len(paraphrased_results)}, n={n}"

        return paraphrased_results


class DifferentFormatsPerQuestionSampler(FormatSampler):
    """
    Instead of repeating exactly the same format response, we sample multiple formats per question
    up to 10
    """

    def __init__(
        self,
        n_formats_per_question: int,
        formatter_options: FormatterOptions,
        exclude_formatters: Sequence[type[StageOneFormatter]] = [],
    ):
        assert n_formats_per_question >= 1, "n_formats_per_question must be greater than 0"
        assert n_formats_per_question <= 10, "n_formats_per_question must be less or equalthan 10"
        self.n_formats_per_question = n_formats_per_question
        self._exclude_formatters = exclude_formatters

        self._formatter_options = formatter_options
        formatters = match_formatter_options(formatter_options)
        self.cot_formatters = Slist(formatters.cot_formatters).filter(lambda x: x.formatter not in exclude_formatters)
        self.non_cot_formatters = Slist(formatters.non_cot_formatters).filter(
            lambda x: x.formatter not in exclude_formatters
        )

    @property
    def excluded_formatters(self) -> Sequence[type[StageOneFormatter]]:
        return self._exclude_formatters

    @property
    def format_options_name(self) -> str:
        return self._formatter_options.value

    @property
    def eligible_cot_formatters(self) -> Slist[FormatterWithPossibleIntervention]:
        return self.cot_formatters

    @property
    def eligible_non_cot_formatters(self) -> Slist[FormatterWithPossibleIntervention]:
        return self.non_cot_formatters

    def sample(
        self,
        tasks: Sequence[BaseTaskOutput],
        n: int,
        use_formatters: Literal["cot"] | Literal["non_cot"],
    ) -> Slist[BaseTaskOutput]:
        """
        Takes a sequnce of outputs and returns a sequence of outputs of length n
        """
        match use_formatters:
            case "cot":
                formatters = self.cot_formatters
            case "non_cot":
                formatters = self.non_cot_formatters

        # if self.n_formats_per_question > len(formatters):
        #     print(
        #         f"Warning: n_formats_per_question={self.n_formats_per_question} > len(formatters):{len(formatters)}: , using all formatters"
        #     )

        n_formats_per_question = self.n_formats_per_question

        tasks = Slist(tasks)
        n_unique_cots = math.ceil(n / n_formats_per_question)
        print("using n_unique_cots", n_unique_cots)
        tasks = tasks.take(n_unique_cots)

        output: Slist[BaseTaskOutput] = Slist()

        task_sample_map_cot: Mapping[str, Sequence[BaseTaskOutput]] = read_task_sample_map_cot()
        task_sample_map_non_cot: Mapping[str, Sequence[BaseTaskOutput]] = read_task_sample_map_non_cot()

        formatter_counts = Counter()
        for task in tasks:
            retrieved_samples = (
                task_sample_map_cot[task.get_task_spec().get_task_hash()]
                if task.get_task_spec().formatter_name == "ZeroShotCOTUnbiasedFormatter"
                else task_sample_map_non_cot[task.get_task_spec().get_task_hash()]
                if task.get_task_spec().formatter_name == "ZeroShotUnbiasedFormatter"
                else assert_should_not_happen()
            )
            task_samples = Slist(retrieved_samples).sample(n_formats_per_question, seed=task.uid())
            rng = random.Random(task.uid())
            # sample with replacement
            sampled_formatters = rng.choices(formatters, k=n_formats_per_question)

            assert len(task_samples) == len(sampled_formatters), "We do not have enough formatters. Wth?"
            formatter_counts.update(Counter([i.name() for i in sampled_formatters]))

            zipped = task_samples.zip(sampled_formatters)

            for task_sample, formatter in zipped:
                intervention = formatter.intervention
                if intervention is not None:
                    new_messages = intervention.intervene(
                        question=task_sample.get_task_spec().get_data_example_obj(),
                        formatter=formatter.formatter,
                        model=task_sample.get_task_spec().inference_config.model,
                    )
                else:
                    new_messages = formatter.formatter.format_example(
                        task_sample.get_task_spec().get_data_example_obj()
                    )
                new_task = task_sample.update_messages_in_task_spec(messages=new_messages)
                output.append(new_task)

        output = output.take(n)
        assert len(output) == n, f"len(output)={len(output)}, n={n}"
        print(f"Formatter counts:\n{json.dumps(formatter_counts, indent=2)}")

        return output

    def __repr__(self) -> str:
        return f"DifferentFormatsPerQuestionSampler(n_formats_per_question={self.n_formats_per_question})"

    def for_legend(self) -> str:
        return f"different_formats={self.n_formats_per_question}"
