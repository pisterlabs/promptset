# pylint: disable=no-member,wrong-import-position,no-name-in-module,arguments-renamed
import json
from typing import Union

import jsonschema

from instill.clients import InstillClient
from instill.protogen.vdp.pipeline.v1beta.pipeline_pb2 import Component
from instill.resources import const
from instill.resources.connector import Connector
from instill.resources.schema import (
    instill_task_classification_input,
    instill_task_detection_input,
    instill_task_image_to_image_input,
    instill_task_instance_segmentation_input,
    instill_task_keypoint_input,
    instill_task_ocr_input,
    instill_task_semantic_segmentation_input,
    instill_task_text_generation_input,
    instill_task_text_to_image_input,
    instill_task_visual_question_answering_input,
)
from instill.resources.schema.huggingface import HuggingFaceConnectorSpec
from instill.resources.schema.instill import (
    InstillModelConnector as InstillModelConnectorConfig,
)
from instill.resources.schema.openai import OpenAIConnectorResource
from instill.resources.schema.stabilityai import StabilityAIConnectorResource


class HuggingfaceConnector(Connector):
    """Huggingface Connector"""

    with open(
        f"{const.SPEC_PATH}/huggingface_definitions.json", "r", encoding="utf8"
    ) as f:
        definitions_jsonschema = json.loads(f.read())

    def __init__(
        self,
        client: InstillClient,
        name: str,
        config: HuggingFaceConnectorSpec,
    ) -> None:
        definition = "connector-definitions/hugging-face"

        jsonschema.validate(vars(config), StabilityAIConnector.definitions_jsonschema)
        super().__init__(client, name, definition, vars(config))


class InstillModelConnector(Connector):
    """Instill Model Connector"""

    with open(f"{const.SPEC_PATH}/instill_definitions.json", "r", encoding="utf8") as f:
        definitions_jsonschema = json.loads(f.read())

    def __init__(
        self,
        client: InstillClient,
        config: InstillModelConnectorConfig,
        name: str = "model-connector",
    ) -> None:
        definition = "connector-definitions/instill-model"

        if config.api_token == "":  # type: ignore
            config.api_token = client.model_service.hosts[  # type: ignore
                client.model_service.instance
            ].token
        if config.server_url == "":  # type: ignore
            config.server_url = "http://api-gateway:8080"  # type: ignore

        jsonschema.validate(vars(config), InstillModelConnector.definitions_jsonschema)
        super().__init__(client, name, definition, vars(config))

    def create_component(
        self,
        name: str,
        inp: Union[
            instill_task_classification_input.Input,
            instill_task_detection_input.Input,
            instill_task_instance_segmentation_input.Input,
            instill_task_semantic_segmentation_input.Input,
            instill_task_keypoint_input.Input,
            instill_task_ocr_input.Input,
            instill_task_image_to_image_input.Input,
            instill_task_text_generation_input.Input,
            instill_task_text_to_image_input.Input,
            instill_task_visual_question_answering_input.Input,
        ],
    ) -> Component:
        if isinstance(inp, instill_task_classification_input.Input):
            config = {
                "input": vars(inp),
                "task": "TASK_CLASSIFICATION",
            }
        if isinstance(inp, instill_task_detection_input.Input):
            config = {
                "input": vars(inp),
                "task": "TASK_DETECTION",
            }
        if isinstance(inp, instill_task_instance_segmentation_input.Input):
            config = {
                "input": vars(inp),
                "task": "TASK_INSTANCE_SEGMENTATION",
            }
        if isinstance(inp, instill_task_semantic_segmentation_input.Input):
            config = {
                "input": vars(inp),
                "task": "TASK_SEMANTIC_SEGMENTATION",
            }
        if isinstance(inp, instill_task_keypoint_input.Input):
            config = {
                "input": vars(inp),
                "task": "TASK_KEYPOINT",
            }
        if isinstance(inp, instill_task_ocr_input.Input):
            config = {
                "input": vars(inp),
                "task": "TASK_OCR",
            }
        if isinstance(inp, instill_task_image_to_image_input.Input):
            config = {
                "input": vars(inp),
                "task": "TASK_IMAGE_TO_IMAGE",
            }
        if isinstance(inp, instill_task_text_generation_input.Input):
            config = {
                "input": vars(inp),
                "task": "TASK_TEXT_GENERATION",
            }
        if isinstance(inp, instill_task_text_to_image_input.Input):
            config = {
                "input": vars(inp),
                "task": "TASK_TEXT_TO_IMAGE",
            }
        if isinstance(inp, instill_task_visual_question_answering_input.Input):
            config = {
                "input": vars(inp),
                "task": "TASK_VISUAL_QUESTION_ANSWERING",
            }
        return super()._create_component(name, config)


class StabilityAIConnector(Connector):
    """Stability AI Connector"""

    with open(
        f"{const.SPEC_PATH}/stabilityai_definitions.json", "r", encoding="utf8"
    ) as f:
        definitions_jsonschema = json.loads(f.read())

    def __init__(
        self,
        client: InstillClient,
        name: str,
        config: StabilityAIConnectorResource,
    ) -> None:
        definition = "connector-definitions/stability-ai"

        jsonschema.validate(vars(config), StabilityAIConnector.definitions_jsonschema)
        super().__init__(client, name, definition, vars(config))


class OpenAIConnector(Connector):
    """OpenAI Connector"""

    with open(f"{const.SPEC_PATH}/openai_definitions.json", "r", encoding="utf8") as f:
        definitions_jsonschema = json.loads(f.read())

    def __init__(
        self,
        client: InstillClient,
        name: str,
        config: OpenAIConnectorResource,
    ) -> None:
        definition = "connector-definitions/openai"

        jsonschema.validate(vars(config), OpenAIConnector.definitions_jsonschema)
        super().__init__(client, name, definition, vars(config))
