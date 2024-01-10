#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#

from opentutor_classifier import (
    AnswerClassifier,
    ClassifierConfig,
    AnswerClassifierInput,
    AnswerClassifierResult,
)
import traceback
import asyncio
from opentutor_classifier.lr2.predict import LRAnswerClassifier
from opentutor_classifier.openai.predict import OpenAIAnswerClassifier
from typing import Dict, Any, Tuple, Union, cast
from opentutor_classifier.logger import get_logger


log = get_logger()


class CompositeAnswerClassifier(AnswerClassifier):

    lr_classifier: AnswerClassifier = LRAnswerClassifier()
    openai_classifier: AnswerClassifier = OpenAIAnswerClassifier()

    async def run_lr_evaluate(
        self, answer: AnswerClassifierInput
    ) -> AnswerClassifierResult:
        result = await self.lr_classifier.evaluate(answer)
        return result

    async def run_openai_evaluate(
        self, answer: AnswerClassifierInput
    ) -> AnswerClassifierResult:
        try:
            result = await self.openai_classifier.evaluate(answer)
            return result
        except BaseException as e:
            raise e

    def configure(self, config: ClassifierConfig) -> "AnswerClassifier":
        self.lr_classifier = self.lr_classifier.configure(config)
        self.openai_classifier = self.openai_classifier.configure(config)
        return self

    async def evaluate(self, answer: AnswerClassifierInput) -> AnswerClassifierResult:
        # lr_task = asyncio.wait_for(self.run_lr_evaluate(answer), timeout=20)
        openai_task = asyncio.wait_for(self.run_openai_evaluate(answer), timeout=10.0)
        lr_task = self.run_lr_evaluate(answer)
        results: Tuple[
            Union[AnswerClassifierResult, BaseException],
            Union[AnswerClassifierResult, BaseException],
        ] = await asyncio.gather(lr_task, openai_task, return_exceptions=True)

        if isinstance(results[0], BaseException):
            log.info("lr classifier returned exception:")
            traceback.print_exception(
                BaseException, results[0], results[0].__traceback__
            )

        if isinstance(results[1], BaseException):
            log.info("openai returned exception:")
            traceback.print_exception(
                BaseException, results[1], results[1].__traceback__
            )

        if not isinstance(results[1], AnswerClassifierResult):
            print("returning LR2 results")
            print(str(cast(AnswerClassifierResult, results[0]).to_dict()))
            return cast(AnswerClassifierResult, results[0])
        else:
            print("returning openai results")
            print(str(cast(AnswerClassifierResult, results[1]).to_dict()))
            return cast(AnswerClassifierResult, results[1])

    def get_last_trained_at(self) -> float:
        return self.lr_classifier.get_last_trained_at()

    def save_config_and_model(self) -> Dict[str, Any]:
        return self.lr_classifier.save_config_and_model()
