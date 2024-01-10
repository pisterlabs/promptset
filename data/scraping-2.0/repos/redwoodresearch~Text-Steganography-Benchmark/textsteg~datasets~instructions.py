import itertools
from typing import Optional

from alpaca_eval.annotators import PairwiseAnnotator
from alpaca_eval.utils import load_or_convert_to_dataframe
from attrs import define
from datasets import load_dataset

from textsteg.hf import generate, hf_models
from textsteg.interface import AndMetadata, Answer, Dataset, Evaluator, Example, Question, TaskImpl
from textsteg.llm import OpenAIChatModel, gpt_3_5, run_llm
from textsteg.log import EntropyProcessor
from textsteg.stats import zip_lists
from textsteg.utils import batchify_and_parrarelize, maybe_range, suppress_tqdm


@define
class InstructionDataset(Dataset, Evaluator):
    """Instruction dataset.

    A "question" is a instruction (+input), and "answers" are the outputs."""

    train_set: list[Example] = []
    val_set: list[Example] = []
    prompt_to_ref: dict[Question, Answer] = {}
    prompt_to_instruction_input: dict[Question, tuple[str, str]] = {}
    default_answer: Answer = "Sorry, I can't help you with that."
    default_answerer_is_ref = True
    annotators_config: str = "chatgpt_fn"

    def __post_init__(self):
        assert all(q in self.prompt_to_ref for q, _ in self.val_set)

    @classmethod
    def alpaca_eval(cls, **kwargs):
        return cls.from_dataset("tatsu-lab/alpaca_eval", **kwargs, val_frac=0.5)

    @classmethod
    def alpaca(cls, **kwargs):
        return cls.from_dataset("tatsu-lab/alpaca", **kwargs, val_frac=0.5)

    @classmethod
    def from_dataset(
        cls,
        dataset_name: str,
        max_train_size: Optional[int] = None,
        max_val_size: Optional[int] = None,
        val_frac: Optional[float] = None,
        seed: int = 0,
        **kwargs,
    ):
        dataset = load_dataset(*dataset_name.split(":"))

        if val_frac is None:
            use_train = max_train_size is None or max_train_size > 0
            if use_train:
                train_ds = (
                    dataset["train"].shuffle(seed=seed).select(maybe_range(len(dataset["train"]), max_train_size))
                )
            else:
                train_ds = []
            val_ds = (
                dataset["validation"].shuffle(seed=seed).select(maybe_range(len(dataset["validation"]), max_val_size))
            )
        else:
            key = next(k for k in ["test", "eval", "val", "validation", "train"] if k in dataset)
            dataset = dataset[key].shuffle(seed=seed)
            val_size = int(len(dataset) * val_frac)
            train_size = len(dataset) - val_size
            train_range = range(min(train_size, max_train_size)) if max_train_size is not None else range(train_size)
            val_range = (
                range(train_size, train_size + min(val_size, max_val_size))
                if max_val_size is not None
                else range(val_size)
            )
            train_ds, val_ds = dataset.select(train_range), dataset.select(val_range)

        prompt_to_ref = {from_ex(ex): ex["output"] for ex in itertools.chain(train_ds, val_ds)}
        prompt_to_instruction_input = {
            from_ex(ex): (ex["instruction"], ex.get("input")) for ex in itertools.chain(train_ds, val_ds)
        }

        def ds_to_set(ds):
            return [(from_ex(ex), ex["output"]) for ex in ds if from_ex(ex) in prompt_to_ref]

        return cls(ds_to_set(train_ds), ds_to_set(val_ds), prompt_to_ref, prompt_to_instruction_input, **kwargs)

    def evaluate(self, examples: list[Example]) -> AndMetadata[list[float]]:
        example_comparisons = [
            ExampleComparison(*self.prompt_to_instruction_input[prompt], output, self.prompt_to_ref[prompt])
            for prompt, output in examples
        ]
        with suppress_tqdm():
            return evaluate_comparisons(example_comparisons, annotators_config=self.annotators_config), {
                "name": "InstructionDataset-" + self.annotators_config
            }

    def get_ref_task_impl(self):
        def ref_answerer(prompts: list[Question]) -> list[Answer]:
            return [self.prompt_to_ref[prompt] for prompt in prompts], {}

        return ref_answerer

    def get_llama_task_impl(self, *args, **kwargs):
        return get_llama_task_impl(*args, **kwargs)

    def get_oai_task_impl(self, *args, **kwargs):
        return get_oai_task_impl(*args, default_answer=self.default_answer, **kwargs)


def get_llama_task_impl(model_name="NousResearch/Llama-2-7b-chat-hf", temperature=0.8, batch_size=8):
    from transformers import LogitsProcessorList, PreTrainedModel, PreTrainedTokenizerFast

    models_and_tokenizers = zip_lists(*hf_models(model_name))

    def llama_answerer(
        prompts: list[Question], model_and_tokenizer: tuple[PreTrainedModel, PreTrainedTokenizerFast]
    ) -> list[Answer]:
        """Return the answer and the entropy of the logits."""

        prompts = [LLAMA_TEMPLATE.format(question=prompt) for prompt in prompts]

        model, tokenizer = model_and_tokenizer

        entropy_processor = EntropyProcessor()
        processors = LogitsProcessorList([entropy_processor])

        generations = generate(
            model,
            tokenizer,
            prompts,
            temperature=temperature,
            logits_processor=processors,
            cache_model_name=model_name,
        )
        entropies = entropy_processor.clean_and_aggr()

        return zip_lists(generations, entropies)

    llama_answerer_batched = batchify_and_parrarelize(
        llama_answerer, models_and_tokenizers, batch_size=batch_size, desc="Following instructions"
    )

    def r(prompts: list[Question]) -> AndMetadata[list[Answer]]:
        answers_and_entropies = llama_answerer_batched(prompts)
        return [a for a, _ in answers_and_entropies], {"entropies": [e for _, e in answers_and_entropies]}

    return r


def get_oai_task_impl(llm: OpenAIChatModel = gpt_3_5(), default_answer: Answer = "") -> TaskImpl:
    def llm_answerer(prompts: list[Question]) -> list[Answer]:
        prompts = [[("user", prompt)] for prompt in prompts]
        responses = run_llm(llm, prompts, desc="Following instructions")
        answers = [response[0].completion if response else default_answer for response in responses]
        return answers, {}

    return llm_answerer


@define
class ExampleComparison:
    instruction: str
    input: Optional[str]
    output: Answer
    ref_output: Answer

    def _to_dict(self, output: Answer, max_output_size: Optional[int] = None):
        if max_output_size is not None:
            output = output[:max_output_size]

        r = {"instruction": self.instruction, "output": output}
        if self.input:
            r["input"] = self.input
        return r

    def to_output_dict(self, max_output_size=None):
        return self._to_dict(self.output, max_output_size=max_output_size)

    def to_ref_output_dict(self, max_output_size=None):
        return self._to_dict(self.ref_output, max_output_size=max_output_size)


LLAMA_TEMPLATE = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{question} [/INST]"""


def from_ex(ex) -> Question:
    instruction = ex["instruction"]
    if "input" in ex and ex["input"]:
        return f"{instruction}\n\n{ex['input']}"
    else:
        return instruction


def evaluate_comparisons(
    examples: list[ExampleComparison], annotators_config="chatgpt_fn", annotator_kwargs={}, annotation_kwargs={}
) -> list[float]:
    max_output_size = {"chatgpt_fn": 3000, "alpaca_eval_gpt4": 4000}.get(annotators_config)
    model_outputs = load_or_convert_to_dataframe([e.to_output_dict(max_output_size) for e in examples])
    reference_outputs = load_or_convert_to_dataframe([e.to_ref_output_dict(max_output_size) for e in examples])
    annotator = PairwiseAnnotator(annotators_config=annotators_config, **annotator_kwargs)
    annotations = annotator.annotate_head2head(
        outputs_1=reference_outputs, outputs_2=model_outputs, **annotation_kwargs
    )
    preference_to_score = {
        1: 0,
        2: 1,
        0: 0.5,
        None: 0.5005,  # When debugging, it's useful to have a score that's not exactly the same as with a real tie
    }
    return [preference_to_score[annotation["preference"]] for annotation in annotations]


if __name__ == "__main__":
    annotator = "alpaca_eval_gpt4"
    example_comparisons = [
        ExampleComparison(
            "I'm trying to find a place to eat in the city. I want to eat at a restaurant that serves Italian food. Can you help me find a place?",
            None,
            "I'm not sure what you mean by that.",
            "Sure, I can help you find a place to eat. What kind of food do you want to eat?",
        ),
    ]
    print(evaluate_comparisons(example_comparisons, annotators_config=annotator))
