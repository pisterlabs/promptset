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
    ExpectationClassifierResult,
    ExpectationConfig,
    ARCH_OPENAI_CLASSIFIER,
)
from opentutor_classifier.dao import ModelRef, find_predicton_config_and_pickle
from opentutor_classifier.openai.shared import OpenAIGroundTruth
from opentutor_classifier.speechact import SpeechActClassifier
from opentutor_classifier.config import EVALUATION_GOOD, EVALUATION_BAD
from opentutor_classifier.openai.openai_api import (
    Answer,
    OpenAICall,
    OpenAIResultContent,
    openai_create,
)
from .constants import (
    SYSTEM_ASSIGNMENT,
    USER_GUARDRAILS,
    ANSWER_TEMPLATE,
    GROUNDTRUTH_FILENAME,
)
from typing import Dict, List, Any


class OpenAIAnswerClassifier(AnswerClassifier):

    speech_act_classifier: SpeechActClassifier = SpeechActClassifier()
    config: ClassifierConfig

    def configure(self, config: ClassifierConfig) -> "AnswerClassifier":
        self.config = config
        return self

    async def evaluate(self, answer: AnswerClassifierInput) -> AnswerClassifierResult:
        if answer.config_data is None:
            raise Exception("missing question data in answer")
        model_ref = ModelRef(
            ARCH_OPENAI_CLASSIFIER, self.config.model_name, GROUNDTRUTH_FILENAME
        )
        ground_truth_dict = find_predicton_config_and_pickle(
            model_ref, self.config.dao
        ).model
        if ground_truth_dict is not None:
            ground_truth = OpenAIGroundTruth.from_dict(ground_truth_dict)
        else:
            ground_truth = None
        concepts: List[ExpectationConfig] = answer.config_data.expectations
        call = OpenAICall(
            system_assignment=SYSTEM_ASSIGNMENT,
            user_concepts=concepts,
            user_answer=[answer.input_sentence],
            user_template=ANSWER_TEMPLATE,
            user_guardrails=USER_GUARDRAILS,
            user_groundtruth=ground_truth,
        )
        response: OpenAIResultContent = await openai_create(call_data=call)
        expectations: List[ExpectationClassifierResult] = []
        print(response.to_json())
        open_ai_answer: Answer = response.answers[
            response.answers.__iter__().__next__()
        ]
        for concept_key in open_ai_answer.concepts.keys():
            concept = open_ai_answer.concepts[concept_key]
            evaluation = EVALUATION_GOOD if concept.is_known else EVALUATION_BAD
            concept_result = ExpectationClassifierResult(
                expectation_id=concept_key,
                evaluation=evaluation,
                score=concept.confidence,
            )
            expectations.append(concept_result)
        result = AnswerClassifierResult(input=answer, expectation_results=expectations)

        result.speech_acts[
            "metacognitive"
        ] = self.speech_act_classifier.check_meta_cognitive(result)
        result.speech_acts["profanity"] = self.speech_act_classifier.check_profanity(
            result
        )

        return result

    def get_last_trained_at(self) -> float:
        return 0.0

    def save_config_and_model(self) -> Dict[str, Any]:
        raise NotImplementedError()
