import sys
from abc import ABC, abstractmethod
from time import sleep
from typing import Collection

import openai
import requests
from openai.error import OpenAIError

from data import AnswerPrediction, Article, Item, Logprobs
from prompting import PickAnswerPrompter, TrueFalseAnswerPrompter

openai.api_key_path = "openai.key"


class QAModel(ABC):
    @abstractmethod
    def answer(self, text: str | None, item: Item) -> list[AnswerPrediction]:
        raise NotImplementedError()

    @classmethod
    def get_subclasses(cls) -> list[type["QAModel"]]:
        subclasses = []
        for subclass in cls.__subclasses__():
            subclasses.append(subclass)
            subclasses.extend(subclass.get_subclasses())
        return subclasses


class MajorityBaseline(QAModel):
    def __init__(self, train_file: str):
        with open(train_file) as f:
            articles = [Article.from_json(line) for line in f]
        true_count = 0
        false_count = 0
        for article in articles:
            for item in article.items:
                for answer in item.answers:
                    if answer.correct:
                        true_count += 1
                    else:
                        false_count += 1
        self.pred_correct = true_count > false_count

    def answer(self, text: str | None, item: Item) -> list[AnswerPrediction]:
        return [
            AnswerPrediction(pred_correct=self.pred_correct) for _ in item.answers
        ]


class ExactMatcher(QAModel):
    def answer(self, text: str, item: Item) -> list[AnswerPrediction]:
        assert text is not None, "ExactMatcher does not support guessing."
        answer_predictions = []
        for answer in item.answers:
            pred_correct = answer.text in text
            answer_prediction = AnswerPrediction(pred_correct=pred_correct)
            answer_predictions.append(answer_prediction)
        return answer_predictions


class FuzzyMatcher(QAModel):
    def __init__(self, multi: bool = False, threshold: float = 0.8, case_insensitive: bool = False):
        self.multi = multi
        self.threshold = threshold
        self.case_insensitive = case_insensitive

    def _shortest_substring_distance(self, substring: str, string: str) -> int:
        dp = [[None for _ in range(len(substring) + 1)] for _ in range(len(string) + 1)]
        for i in range(len(string) + 1):
            for j in range(len(substring) + 1):
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = 0
                else:
                    add_cost = 1 if j < len(substring) else 0
                    del_cost = 1
                    sub_cost = 1 if string[i - 1] != substring[j - 1] else 0
                    dp[i][j] = min(
                        dp[i - 1][j] + add_cost,
                        dp[i][j - 1] + del_cost,
                        dp[i - 1][j - 1] + sub_cost,
                    )
        return dp[-1][-1]

    def answer(self, text: str, item: Item) -> list[AnswerPrediction]:
        assert text is not None, "FuzzyMatcher does not support guessing."
        if self.case_insensitive:
            text = text.casefold()
        answer_predictions = []
        best_answer_index = None
        max_prob = -1
        for i, answer in enumerate(item.answers):
            answer_text = answer.text
            if self.case_insensitive:
                answer_text = answer_text.casefold()
            distance = self._shortest_substring_distance(answer_text, text)
            prob_correct = 1 - distance / len(answer_text)
            if prob_correct > max_prob:
                best_answer_index = i
                max_prob = prob_correct
            pred_correct = prob_correct >= self.threshold
            answer_prediction = AnswerPrediction(
                pred_correct=pred_correct, prob_correct=prob_correct
            )
            answer_predictions.append(answer_prediction)
        if not self.multi:
            for i, answer_prediction in enumerate(answer_predictions):
                if i == best_answer_index:
                    answer_prediction.pred_correct = True
                else:
                    answer_prediction.pred_correct = False
        return answer_predictions


class GenerativeModel(QAModel):
    def __init__(
        self,
        *,
        lang: str,
        chat: bool,
        true_false: bool,
        multi: bool | None = None,
    ):
        if true_false and not multi:
            raise ValueError("true_false is only supported for multi=True.")
        self.lang = lang
        self.chat = chat
        self.true_false = true_false
        self.multi = multi

    @abstractmethod
    def _generate(
        self, prompt: str, labels: Collection[str]
    ) -> tuple[str, Logprobs | None]:
        raise NotImplementedError()

    def answer(self, text: str | None, item: Item) -> list[AnswerPrediction]:
        guess = text is None
        answer_predictions = []

        if self.true_false:  # Predict each answer separately
            prompter = TrueFalseAnswerPrompter(
                lang=self.lang, chat=self.chat, guess=guess
            )
            for answer in item.answers:
                prompt = prompter.build_prompt(text, item.question, answer.text)
                model_output, logprobs = self._generate(
                    prompt, [prompter.true_label, prompter.false_label]
                )
                pred_correct, prob_correct = prompter.parse_output(model_output, logprobs)
                answer_prediction = AnswerPrediction(
                    pred_correct=pred_correct,
                    prob_correct=prob_correct,
                    model_output=model_output,
                    logprobs=logprobs,
                )
                answer_predictions.append(answer_prediction)

        else:  # Predict all answers at once
            multi = item.multiple if self.multi is None else self.multi
            prompter = PickAnswerPrompter(
                lang=self.lang, chat=self.chat, guess=guess, multi=multi
            )
            prompt = prompter.build_prompt(
                text, item.question, [answer.text for answer in item.answers]
            )
            model_output, logprobs = self._generate(
                prompt, prompter.get_labels(len(item.answers))
            )
            preds_correct, probs_correct = (
                prompter.parse_output(model_output, logprobs, len(item.answers))
            )
            for pred_correct, prob_correct in zip(preds_correct, probs_correct):
                answer_prediction = AnswerPrediction(
                    pred_correct=pred_correct,
                    prob_correct=prob_correct,
                    model_output=model_output,
                    logprobs=logprobs,
                )
                answer_predictions.append(answer_prediction)

        return answer_predictions


class OpenAIModel(GenerativeModel):
    def __init__(self, model: str, max_retries: int = 10, retry_wait_seconds: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.max_retries = max_retries
        self.retry_wait_seconds = retry_wait_seconds

    def _generate(
        self, prompt: str, labels: Collection[str]
    ) -> tuple[str, Logprobs | None]:
        for _ in range(self.max_retries):
            try:
                if self.chat:
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=10,
                    )
                    print(
                        "Tokens used:",
                        response.usage.prompt_tokens,
                        "+",
                        response.usage.completion_tokens,
                        file=sys.stderr,
                    )
                    model_output = response.choices[0].message.content
                    logprobs = None  # Chat completion does not support logprobs (yet)
                else:
                    response = openai.Completion.create(
                        model=self.model,
                        prompt=prompt,
                        temperature=0.0,
                        max_tokens=10,
                        logprobs=10,
                    )
                    print(
                        "Tokens used:",
                        response.usage.prompt_tokens,
                        "+",
                        response.usage.completion_tokens,
                        file=sys.stderr,
                    )
                    model_output = response.choices[0].text
                    logprobs = response.choices[0].logprobs.top_logprobs
                return model_output, logprobs

            except OpenAIError as e:
                print(
                    f"Error while querying OpenAI API: {e}\nRetrying in {self.retry_wait_seconds} second(s)...",
                    file=sys.stderr,
                )
                sleep(self.retry_wait_seconds)

        raise RuntimeError(f"{self.max_retries} errors in a row, aborting.")


class LLMAPIModel(GenerativeModel):
    def __init__(self, base_url: str = "http://localhost:8000", **kwargs):
        super().__init__(**kwargs)
        self.base_url = base_url

    def _generate(
        self, prompt: str, labels: Collection[str]
    ) -> tuple[str, Logprobs | None]:
        logprobs_for_tokens = list(labels)
        if self.chat:
            response = requests.post(
                f"{self.base_url}/chat",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 10,
                    "top_logprobs": 1,
                    "logprobs_for_tokens": logprobs_for_tokens,
                    "config": {
                        "temperature": 0.0,
                    },
                },
            ).json()
        else:
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": 10,
                    "top_logprobs": 1,
                    "logprobs_for_tokens": logprobs_for_tokens,
                    "config": {
                        "temperature": 0.0,
                    },
                },
            ).json()

        model_output = response["output"]
        logprobs = response["logprobs"]
        return model_output, logprobs


MODEL_CLASSES: dict[str, type[QAModel]] = {
    subclass.__name__: subclass
    for subclass in QAModel.get_subclasses()
    if not subclass.__abstractmethods__
}
