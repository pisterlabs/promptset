from __future__ import annotations
import copy
import json
import logging
import os
from typing import Any, List, Text, Dict, Type, Union, Tuple, Optional

import openai
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import DEFAULT_NLU_FALLBACK_INTENT_NAME
from rasa.core.constants import (
    DEFAULT_NLU_FALLBACK_THRESHOLD,
    DEFAULT_NLU_FALLBACK_AMBIGUITY_THRESHOLD,
)
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import (
    INTENT,
    INTENT_NAME_KEY,
    INTENT_RANKING_KEY,
    PREDICTED_CONFIDENCE_KEY,
)

TEXT_KEY = "text"
NAME_KEY = "name"
MODEL_KEY = "model"
TEMPERATURE_KEY = "temperature"
MAX_TOKENS_KEY = "max_tokens"
THRESHOLD_KEY = "threshold"
AMBIGUITY_THRESHOLD_KEY = "ambiguity_threshold"

DEFAULT_MODEL = "text-davinci-003"
DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_TOKENS = 1000

logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=False
)
class LLMFallbackReClassifier(GraphComponent, IntentClassifier):
    """Handles incoming messages with low NLU confidence using LLMs."""

    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        return [IntentClassifier]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """The component's default config"""
        return {
            MODEL_KEY: DEFAULT_MODEL,
            TEMPERATURE_KEY: DEFAULT_TEMPERATURE,
            MAX_TOKENS_KEY: DEFAULT_MAX_TOKENS,
            # If all intent confidence scores are beyond this threshold, set the current
            # intent to `FALLBACK_INTENT_NAME`
            THRESHOLD_KEY: DEFAULT_NLU_FALLBACK_THRESHOLD,
            # If the confidence scores for the top two intent predictions are closer
            # than `AMBIGUITY_THRESHOLD_KEY`,
            # then `FALLBACK_INTENT_NAME` is predicted.
            AMBIGUITY_THRESHOLD_KEY: DEFAULT_NLU_FALLBACK_AMBIGUITY_THRESHOLD,
        }

    def __init__(self, config: Dict[Text, Any]) -> None:
        """Constructs a new LLM fallback re-classifier."""
        self.component_config = config

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> LLMFallbackReClassifier:
        """Creates a new component."""
        return cls(config)

    def process(self, messages: List[Message]) -> List[Message]:
        """Process a list of incoming messages.

        This is the component's chance to process incoming
        messages. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`rasa.nlu.components.Component.create`
        of ANY component and
        on any context attributes created by a call to
        :meth:`rasa.nlu.components.Component.process`
        of components previous to this one.

        Args:
            messages: List containing :class:
            `rasa.shared.nlu.training_data.message.Message` to process.
        """
        for message in messages:
            if not self._should_fallback_re_classify(message):
                continue

            # reaching here means a nlu_fallback is triggered
            # however, the following code overrides the default
            # behaviour and hands over the final decision to the
            # LLM.
            logger.info(
                f"The intent is a `nlu_fallback`. However, the "
                f"decision will be handed over to the LLM."
            )

            # logging the initial intent ranking list
            logger.debug("Initial intent ranking:")
            _log_intent_ranking(intent_ranking=message.data[INTENT_RANKING_KEY])

            text = message.data.get(TEXT_KEY, "")
            intents = [
                intent.get(NAME_KEY)
                for intent in message.data.get(INTENT_RANKING_KEY, [])
            ]

            # LLM will consider the query and the top ranking intents
            # and attempt to re-classify the query under one of those,
            # including `nlu_fallback` intent.
            llm_classification = _get_llm_intent_classification(
                text=text, intents=intents, configs=self.component_config
            )

            # we assume that the fallback confidence
            # is the same as the fallback threshold
            confidence = self.component_config[THRESHOLD_KEY]
            message.data.setdefault(INTENT_RANKING_KEY, [])

            if llm_classification == DEFAULT_NLU_FALLBACK_INTENT_NAME:
                message.data[INTENT] = _fallback_intent(confidence)
                message.data[INTENT_RANKING_KEY].insert(0, _fallback_intent(confidence))
            else:
                message.data[INTENT] = _fallback_re_classified_intent(
                    intent=llm_classification, confidence=confidence
                )
                # remove the current ranking for
                # the llm predicted intent
                message.data[INTENT_RANKING_KEY] = [
                    ranking
                    for ranking in message.data[INTENT_RANKING_KEY]
                    if ranking.get("name") not in [llm_classification]
                ]
                # append the new ranking predicted by the llm
                message.data[INTENT_RANKING_KEY].insert(
                    0,
                    _fallback_re_classified_intent(
                        intent=llm_classification, confidence=confidence
                    ),
                )

                # logging the new intent ranking list
                logger.debug("New intent ranking:")
                _log_intent_ranking(intent_ranking=message.data[INTENT_RANKING_KEY])

        return messages

    def _should_fallback_re_classify(self, message: Message) -> bool:
        """Check if the fallback intent can be re-classified.

        Args:
            message: The current message and its intent predictions.

        Returns:
            `True` if the fallback intent should be predicted.
        """
        intent_name = message.data[INTENT].get(INTENT_NAME_KEY)
        below_threshold, nlu_confidence = self._nlu_confidence_below_threshold(message)

        if below_threshold:
            logger.debug(
                f"NLU confidence {nlu_confidence} for intent '{intent_name}' is lower "
                f"than NLU threshold {self.component_config[THRESHOLD_KEY]:.2f}."
            )
            return True

        ambiguous_prediction, confidence_delta = self._nlu_prediction_ambiguous(message)
        if ambiguous_prediction:
            logger.debug(
                f"The difference in NLU confidences "
                f"for the top two intents ({confidence_delta}) is lower than "
                f"the ambiguity threshold "
                f"{self.component_config[AMBIGUITY_THRESHOLD_KEY]:.2f}. Predicting "
                f"intent '{DEFAULT_NLU_FALLBACK_INTENT_NAME}' instead of "
                f"'{intent_name}'."
            )
            return True

        return False

    def _nlu_confidence_below_threshold(self, message: Message) -> Tuple[bool, float]:
        nlu_confidence = message.data[INTENT].get(PREDICTED_CONFIDENCE_KEY)
        return nlu_confidence < self.component_config[THRESHOLD_KEY], nlu_confidence

    def _nlu_prediction_ambiguous(
        self, message: Message
    ) -> Tuple[bool, Optional[float]]:
        intents = message.data.get(INTENT_RANKING_KEY, [])
        if len(intents) >= 2:
            first_confidence = intents[0].get(PREDICTED_CONFIDENCE_KEY, 1.0)
            second_confidence = intents[1].get(PREDICTED_CONFIDENCE_KEY, 1.0)
            difference = first_confidence - second_confidence
            return (
                difference < self.component_config[AMBIGUITY_THRESHOLD_KEY],
                difference,
            )
        return False, None


def _fallback_intent(confidence: float) -> Dict[Text, Union[Text, float]]:
    return {
        INTENT_NAME_KEY: DEFAULT_NLU_FALLBACK_INTENT_NAME,
        PREDICTED_CONFIDENCE_KEY: confidence,
    }


def _fallback_re_classified_intent(
    intent: Text, confidence: float
) -> Dict[Text, Union[Text, float]]:
    return {
        INTENT_NAME_KEY: intent,
        PREDICTED_CONFIDENCE_KEY: confidence,
    }


def is_llm_fallback_re_classifier_prediction(prediction: Dict[Text, Any]) -> bool:
    """Checks if the intent was predicted by the `LLMFallbackReClassifier`.

    Args:
        prediction: The prediction of the NLU model.

    Returns:
        `True` if the top classified intent was the fallback intent.
    """
    return (
        prediction.get(INTENT, {}).get(INTENT_NAME_KEY)
        == DEFAULT_NLU_FALLBACK_INTENT_NAME
    )


def undo_llm_fallback_prediction(prediction: Dict[Text, Any]) -> Dict[Text, Any]:
    """Undo the prediction of the fallback intent.

    Args:
        prediction: The prediction of the NLU model.

    Returns:
        The prediction as if the `LLMFallbackReClassifier` wasn't present in the pipeline.
        If the fallback intent is the only intent, return the prediction as it was
        provided.
    """
    intent_ranking = prediction.get(INTENT_RANKING_KEY, [])
    if len(intent_ranking) < 2:
        return prediction

    prediction = copy.deepcopy(prediction)
    prediction[INTENT] = intent_ranking[1]
    prediction[INTENT_RANKING_KEY] = prediction[INTENT_RANKING_KEY][1:]

    return prediction


def _get_llm_intent_classification(text: Text, intents: List, configs: Dict) -> Text:
    try:
        prompt_template = """
You are tasked with classifying a given query under one of several classes. The goal is to create a JSON response that contains the classified intent. The JSON should be in the format: {"intent": "classified_class"}. If the query does not fit any of the classes provided, the intent should be set as "nlu_fallback", indicating that the query cannot be categorized under the given set of classes. Remember, you must only use the provided classes and should not introduce any new ones.

Instructions:

1. Take the input query and the list of classes.
2. Determine the class that best represents the query.
3. If the query fits one of the provided classes, construct a JSON response with the classified intent in the following format 

{"intent": "classified_class"}

4. If the query does not fit any of the provided classes, construct a JSON response with the intent set as "nlu_fallback".
5. Provide only the JSON response.

query: {{text}}
classes: [{{intents}}]"""

        # construct the prompt from template
        intent_str = ", ".join(intents)
        prompt = prompt_template.replace("{{text}}", text)
        prompt = prompt.replace("{{intents}}", intent_str)

        # get the LLM completion
        openai.api_key = os.getenv("OPENAI_COMPLETIONS_API_KEY")

        response = openai.Completion.create(
            model=configs[MODEL_KEY],
            prompt=prompt,
            temperature=configs[TEMPERATURE_KEY],
            max_tokens=configs[MAX_TOKENS_KEY],
        )

        choices = response.get("choices", [])
        llm_intent = _extract_llm_classification(choices=choices)

        if llm_intent not in intents:
            llm_intent = DEFAULT_NLU_FALLBACK_INTENT_NAME

        if not llm_intent:
            llm_intent = DEFAULT_NLU_FALLBACK_INTENT_NAME

        return llm_intent
    except Exception as e:
        logger.exception(
            f"Error occurred while getting the LLM re-classified intent. "
            f"The intent will be replaced as a `nlu_fallback`. {e}"
        )
        return DEFAULT_NLU_FALLBACK_INTENT_NAME


def _extract_llm_classification(choices: Dict) -> Optional[Text]:
    try:
        choice = choices[0].get("text", "")
        logger.info(f"LLM classification: {choice}")

        response_json = json.loads(choice)
        return response_json.get("intent", "")
    except Exception as e:
        logger.exception(f"Error occurred while extracting the LLM response. {e}")
        return ""


def _log_intent_ranking(intent_ranking: Dict):
    for intent in intent_ranking:
        logger.debug(f"name: {intent.get('name')}")
        logger.debug(f"name: {intent.get('confidence')}")
    logger.debug("")
