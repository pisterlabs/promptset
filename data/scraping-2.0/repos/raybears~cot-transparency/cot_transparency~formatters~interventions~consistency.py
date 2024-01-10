from typing import Optional, Sequence, Type

from cot_transparency.apis.base import Prompt
from cot_transparency.apis.openai import OpenAICompletionPrompt
from cot_transparency.data_models.data.aqua import AquaExample
from cot_transparency.data_models.data.biased_question_unbiased_cot import (
    format_big_brain_question_cot,
)
from cot_transparency.data_models.data.hellaswag import HellaSwagExample
from cot_transparency.data_models.data.inverse_scaling import InverseScalingExample
from cot_transparency.data_models.data.logiqa import LogicQaExample
from cot_transparency.data_models.data.mmlu import MMLUExample
from cot_transparency.data_models.data.truthful_qa import TruthfulQAExample
from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.messages import ChatMessage
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.instructions import UNBIASED_CONTROL_TOKEN
from cot_transparency.formatters.interventions.assistant_completion_utils import (
    insert_to_after_system_message,
    prepend_to_front_first_user_message,
    prepend_to_front_system_message,
)
from cot_transparency.formatters.interventions.big_brain_few_shots_loading import (
    get_big_brain_cots,
)
from cot_transparency.formatters.interventions.few_shots_loading import (
    get_correct_cots,
    get_correct_cots_claude_2,
    get_correct_cots_inverse_scaling_for_task,
    get_correct_cots_testing_by_name,
)
from cot_transparency.formatters.interventions.formatting import (
    format_biased_question_cot,
    format_biased_question_non_cot_random_formatter,
    format_biased_question_non_cot_sycophancy,
    format_few_shot_for_prompt_sen,
    format_pair_cot,
    format_pair_non_cot,
    format_unbiased_question_cot,
)
from cot_transparency.formatters.interventions.intervention import Intervention


class PairedConsistency6(Intervention):
    # Because it is a pair, sample 6 / 2 = 3
    n_samples: int = 3

    @classmethod
    def intervene(
        cls,
        question: DataExampleBase,
        formatter: Type[StageOneFormatter],
        model: Optional[str] = None,
    ) -> Sequence[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            get_correct_cots().sample(cls.n_samples, seed=question.hash()).map(format_pair_cot).sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=OpenAICompletionPrompt.from_prompt(prompt).format(),
        )
        return new


class PairedConsistency12(PairedConsistency6):
    # Because it is a pair, sample 12 / 2 = 6
    n_samples: int = 6


class PairedConsistency10(PairedConsistency6):
    # Because it is a pair, sample 10 / 2 = 5
    n_samples: int = 5


class RepeatedConsistency10(Intervention):
    # Just the naive few shot, but repeated 5 * 2 = 10
    n_samples: int = 5

    @classmethod
    def intervene(
        cls,
        question: DataExampleBase,
        formatter: Type[StageOneFormatter],
        model: Optional[str] = None,
    ) -> Sequence[ChatMessage]:
        messages = formatter.format_example(question)
        cots = get_correct_cots().sample(cls.n_samples, seed=question.hash())
        duplicated = cots + cots
        prompt: Prompt = duplicated.map(format_unbiased_question_cot).sum_or_raise()
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=OpenAICompletionPrompt.from_prompt(prompt).format(),
        )
        return new


class RepeatedConsistency12(RepeatedConsistency10):
    n_samples: int = 6


class BiasedConsistency10(Intervention):
    @classmethod
    def intervene(
        cls,
        question: DataExampleBase,
        formatter: Type[StageOneFormatter],
        model: Optional[str] = None,
    ) -> Sequence[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            # Not a pair so, sample 10
            get_correct_cots()
            .sample(10, seed=question.hash())
            .map(lambda task: format_biased_question_cot(task=task, formatter=formatter))
            .sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=OpenAICompletionPrompt.from_prompt(prompt).format(),
        )
        return new


class BigBrainBiasedConsistency10(Intervention):
    n_samples: int = 10

    @classmethod
    def intervene(
        cls,
        question: DataExampleBase,
        formatter: Type[StageOneFormatter],
        model: Optional[str] = None,
    ) -> Sequence[ChatMessage]:
        messages = formatter.format_example(question)
        # TODO: filter out to not sample the same biased formatter
        prompt: Prompt = (
            # Not a pair so, sample 10
            get_big_brain_cots()
            .sample(cls.n_samples, seed=question.hash())
            .map(format_big_brain_question_cot)
            .sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=OpenAICompletionPrompt.from_prompt(prompt).format(),
        )
        return new


class BigBrainBiasedConsistency12(BigBrainBiasedConsistency10):
    n_samples: int = 12


class BigBrainBiasedConsistencySeparate10(BigBrainBiasedConsistency10):
    """Separate the few shots into messages rather than in the single message"""

    @classmethod
    def intervene(
        cls,
        question: DataExampleBase,
        formatter: Type[StageOneFormatter],
        model: Optional[str] = None,
    ) -> Sequence[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            # Not a pair so, sample 10
            get_big_brain_cots()
            .sample(10, seed=question.hash())
            .map(format_big_brain_question_cot)
            .sum_or_raise()
        )
        new = insert_to_after_system_message(messages=messages, to_insert=prompt.messages)
        return new


class NaiveFewShot1(Intervention):
    # Simply use unbiased few shot
    n_samples: int = 1

    @classmethod
    def intervene(
        cls,
        question: DataExampleBase,
        formatter: Type[StageOneFormatter],
        model: Optional[str] = None,
    ) -> Sequence[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            get_correct_cots()
            .sample(cls.n_samples, seed=question.hash())
            .map(format_unbiased_question_cot)
            .sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=OpenAICompletionPrompt.from_prompt(prompt).format(),
        )
        return new


class NaiveFewShot3InverseScaling(Intervention):
    # Simply use unbiased few shot
    n_samples: int = 3

    @classmethod
    def intervene(
        cls,
        question: DataExampleBase,
        formatter: Type[StageOneFormatter],
        model: Optional[str] = None,
    ) -> Sequence[ChatMessage]:
        if isinstance(question, InverseScalingExample):
            definitely_inverse_example: InverseScalingExample = question
        else:
            raise ValueError(f"Expected InverseScalingExample, got {question}")
        task_hash = definitely_inverse_example.hash()
        messages = formatter.format_example(definitely_inverse_example)
        prompt: Prompt = (
            get_correct_cots_inverse_scaling_for_task(definitely_inverse_example.task)
            .filter(lambda x: x.data_example_hash() != task_hash)
            .sample(cls.n_samples, seed=question.hash())
            .map(format_unbiased_question_cot)
            .sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=OpenAICompletionPrompt.from_prompt(prompt).format() + "\n",
        )
        return new


class NaiveFewShot1InverseScaling(NaiveFewShot3InverseScaling):
    # Simply use unbiased few shot
    n_samples: int = 1


class NaiveFewShot6InverseScaling(NaiveFewShot3InverseScaling):
    # Simply use unbiased few shot
    n_samples: int = 6


class NaiveFewShot3Testing(Intervention):
    # Simply use unbiased few shot
    n_samples: int = 3

    @classmethod
    def intervene(
        cls,
        question: DataExampleBase,
        formatter: Type[StageOneFormatter],
        model: Optional[str] = None,
    ) -> Sequence[ChatMessage]:
        match question:
            case TruthfulQAExample():
                cots = get_correct_cots_testing_by_name("truthful_qa")
            case MMLUExample():
                cots = get_correct_cots_testing_by_name("mmlu")
            case HellaSwagExample():
                cots = get_correct_cots_testing_by_name("hellaswag")
            case LogicQaExample():
                cots = get_correct_cots_testing_by_name("logiqa")
            case _:
                raise ValueError(f"Expected correct cots for testing, got {question}")
        assert cots.length > 0, f"Expected at least one cot for {question}"
        messages = formatter.format_example(question)
        task_hash = question.hash()
        prompt: Prompt = (
            cots.filter(lambda x: x.data_example_hash() != task_hash)
            .sample(cls.n_samples, seed=question.hash())
            .map(format_unbiased_question_cot)
            .sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=OpenAICompletionPrompt.from_prompt(prompt).format() + "\n",
        )
        return new


class NaiveFewShot1Testing(NaiveFewShot3Testing):
    # Simply use unbiased few shot
    n_samples: int = 1


class OnlyAnswerAFewShot3Testing(Intervention):
    # Simply use unbiased few shot
    n_samples: int = 3

    @classmethod
    def intervene(
        cls,
        question: DataExampleBase,
        formatter: Type[StageOneFormatter],
        model: Optional[str] = None,
    ) -> Sequence[ChatMessage]:
        match question:
            case TruthfulQAExample():
                cots = get_correct_cots_testing_by_name("truthful_qa")
            case MMLUExample():
                cots = get_correct_cots_testing_by_name("mmlu")
            case HellaSwagExample():
                cots = get_correct_cots_testing_by_name("hellaswag")
            case LogicQaExample():
                cots = get_correct_cots_testing_by_name("logiqa")
            case AquaExample():
                cots = get_correct_cots_testing_by_name("aqua")
            case _:
                raise ValueError(f"Expected correct cots for testing, got {question}")
        assert cots.length > 0, f"Expected at least one cot for {question}"
        messages = formatter.format_example(question)
        task_hash = question.hash()
        prompt: Prompt = (
            cots.filter(lambda x: x.data_example_hash() != task_hash)
            .filter(lambda x: x.inference_output.parsed_response == "A")
            .sample(cls.n_samples, seed=question.hash())
            .map(format_unbiased_question_cot)
            .sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=OpenAICompletionPrompt.from_prompt(prompt).format() + "\n",
        )
        return new


# get_correct_cots_testing


class NaiveFewShot3(NaiveFewShot1):
    # Simply use unbiased few shot
    n_samples: int = 3


class NaiveFewShot6(NaiveFewShot1):
    # Simply use unbiased few shot
    n_samples: int = 6


class NaiveFewShot5(NaiveFewShot1):
    # Simply use unbiased few shot
    n_samples: int = 5


class NaiveFewShot10(NaiveFewShot1):
    # Simply use unbiased few shot
    n_samples: int = 10


class NaiveFewShot12(NaiveFewShot1):
    # Simply use unbiased few shot
    n_samples: int = 12


class NaiveFewShot16(NaiveFewShot1):
    # Simply use unbiased few shot
    n_samples: int = 16


class NaiveFewShot32(NaiveFewShot1):
    # Simply use unbiased few shot
    n_samples: int = 32


class ClaudeFewShot1(Intervention):
    n_samples: int = 1

    @classmethod
    def intervene(
        cls,
        question: DataExampleBase,
        formatter: Type[StageOneFormatter],
        model: Optional[str] = None,
    ) -> Sequence[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            get_correct_cots_claude_2()
            .sample(cls.n_samples, seed=question.hash())
            .map(format_unbiased_question_cot)
            .sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=OpenAICompletionPrompt.from_prompt(prompt).format(),
        )
        return new


class ClaudeFewShot3(ClaudeFewShot1):
    n_samples: int = 3


class ClaudeFewShot6(ClaudeFewShot1):
    n_samples: int = 6


class ClaudeFewShot10(ClaudeFewShot1):
    n_samples: int = 10


class ClaudeSeparate10(Intervention):
    # Simply use unbiased few shot
    n_samples: int = 10

    @classmethod
    def intervene(
        cls,
        question: DataExampleBase,
        formatter: Type[StageOneFormatter],
        model: Optional[str] = None,
    ) -> Sequence[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            get_correct_cots_claude_2()
            .sample(cls.n_samples, seed=question.hash())
            .map(format_unbiased_question_cot)
            .sum_or_raise()
        )
        new = insert_to_after_system_message(
            messages=messages,
            to_insert=prompt.messages,
        )
        return new


class ClaudeFewShot16(ClaudeFewShot1):
    n_samples: int = 16


class ClaudeFewShot32(ClaudeFewShot1):
    n_samples: int = 16


class NaiveFewShotSeparate10(Intervention):
    # Simply use unbiased few shot
    n_samples: int = 10

    @classmethod
    def intervene(
        cls,
        question: DataExampleBase,
        formatter: Type[StageOneFormatter],
        model: Optional[str] = None,
    ) -> Sequence[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            get_correct_cots()
            .sample(cls.n_samples, seed=question.hash())
            .map(format_unbiased_question_cot)
            .sum_or_raise()
        )
        new = insert_to_after_system_message(
            messages=messages,
            to_insert=prompt.messages,
        )
        return new


class NaiveFewShotLabelOnly1(Intervention):
    # Non cot, only the label
    n_samples: int = 1

    @classmethod
    def intervene(
        cls,
        question: DataExampleBase,
        formatter: Type[StageOneFormatter],
        model: Optional[str] = None,
    ) -> Sequence[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            get_correct_cots()
            .sample(cls.n_samples, seed=question.hash())
            .map(format_few_shot_for_prompt_sen)
            .sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=OpenAICompletionPrompt.from_prompt(prompt).format(),
        )
        return new


class NaiveFewShotLabelOnly3(NaiveFewShotLabelOnly1):
    n_samples: int = 3


class NaiveFewShotLabelOnly6(NaiveFewShotLabelOnly1):
    n_samples: int = 6


class NaiveFewShotLabelOnly10(Intervention):
    n_samples: int = 10


class NaiveFewShotLabelOnly16(NaiveFewShotLabelOnly1):
    n_samples: int = 16


class NaiveFewShotLabelOnly30(NaiveFewShotLabelOnly1):
    n_samples: int = 30


class NaiveFewShotLabelOnly32(NaiveFewShotLabelOnly30):
    n_samples: int = 32


class SycophancyConsistencyLabelOnly10(Intervention):
    n_samples: int = 10

    @classmethod
    def intervene(
        cls,
        question: DataExampleBase,
        formatter: Type[StageOneFormatter],
        model: Optional[str] = None,
    ) -> Sequence[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            get_correct_cots()
            .sample(cls.n_samples, seed=question.hash())
            .map(format_biased_question_non_cot_sycophancy)
            .sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=OpenAICompletionPrompt.from_prompt(prompt).format(),
        )
        return new


class SycoConsistencyLabelOnly30(SycophancyConsistencyLabelOnly10):
    n_samples: int = 30


class BiasedConsistencyLabelOnly10(Intervention):
    n_samples: int = 10

    @classmethod
    def intervene(
        cls,
        question: DataExampleBase,
        formatter: Type[StageOneFormatter],
        model: Optional[str] = None,
    ) -> Sequence[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            get_correct_cots()
            .sample(10, seed=question.hash())
            .map(lambda task: format_biased_question_non_cot_random_formatter(task=task, formatter=formatter))
            .sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=OpenAICompletionPrompt.from_prompt(prompt).format(),
        )
        return new


class BiasedConsistencyLabelOnly20(BiasedConsistencyLabelOnly10):
    n_samples: int = 20


class BiasedConsistencyLabelOnly30(BiasedConsistencyLabelOnly10):
    n_samples: int = 30


class PairedFewShotLabelOnly10(Intervention):
    # Non cot, only the label
    # Because it is a pair, sample 10 / 2 = 5
    n_samples: int = 5

    @classmethod
    def hook(cls, question: DataExampleBase, messages: Sequence[ChatMessage]) -> Sequence[ChatMessage]:
        prompt: Prompt = (
            get_correct_cots().sample(cls.n_samples, seed=question.hash()).map(format_pair_non_cot).sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=OpenAICompletionPrompt.from_prompt(prompt).format(),
        )
        return new


class PairedFewShotLabelOnly30(PairedFewShotLabelOnly10):
    # Non cot, only the label
    # Because it is a pair, sample 30 / 2 = 15
    n_samples: int = 15


class AddUnbiasedControlToken(Intervention):
    @classmethod
    def intervene(
        cls,
        question: DataExampleBase,
        formatter: Type[StageOneFormatter],
        model: Optional[str] = None,
    ) -> Sequence[ChatMessage]:
        formatted = formatter.format_example(question)
        added_system_unbiased: Sequence[ChatMessage] = prepend_to_front_system_message(
            messages=formatted, prepend=f"{UNBIASED_CONTROL_TOKEN} "
        )
        return added_system_unbiased
