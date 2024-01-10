from typing import List

from bert_score import score as bert_score
from bleurt.score import BleurtScorer
from langchain.chains.base import Chain
from langchain.evaluation import load_evaluator
from langchain.evaluation.schema import EvaluatorType
from nltk.translate.bleu_score import sentence_bleu
from rouge_score.rouge_scorer import RougeScorer

from master_thesis.base import BaseMetric


class BERTScoreMetric(BaseMetric):
    def test(self, question: str, answer: str, prediction: str) -> float:
        _, _, fmeasure = bert_score([prediction], [answer], lang="en")

        return fmeasure.item()


class BLEUMetric(BaseMetric):
    def test(self, question: str, answer: str, prediction: str) -> float:
        return sentence_bleu([answer.split()], prediction.split(), weights=[1])


class BLEURTMetric(BaseMetric):
    _bleurt_scorer: BleurtScorer

    def __init__(self, model_path: str = "../models/bleurt_20_d12") -> None:
        super().__init__()

        self._bleurt_scorer = BleurtScorer(model_path)

    def test(self, question: str, answer: str, prediction: str) -> float:
        return self._bleurt_scorer.score(
            references=[answer],
            candidates=[prediction],
        )[0]


class LLMEvalMetric(BaseMetric):
    _evaluator: Chain

    def __init__(self) -> None:
        super().__init__()

        self._evaluator = load_evaluator(EvaluatorType.QA)

    def test(self, question: str, answer: str, prediction: str) -> float:
        return float(
            self._evaluator.evaluate_strings(
                prediction=prediction,
                reference=answer,
                input=question,
            )["score"]
        )


class ROUGEMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()

        self._rouge_scorer = RougeScorer(["rougeL"], use_stemmer=True)

    def test(self, question: str, answer: str, prediction: str) -> float:
        return self._rouge_scorer.score(answer, prediction)["rougeL"].fmeasure
