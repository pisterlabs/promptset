from typing import *

from openai_api_payloads.text_completion import TextCompletionInputs, TextCompletionOutputs
from pydantic import BaseModel, Field, Required

from .common import BaseTaskInputs, BaseTaskOutputs


class TextCompletionTaskInputs(BaseTaskInputs):
    task_inputs: TextCompletionInputs = Field(default=Required)


class TextCompletionTaskOutputs(BaseTaskOutputs):
    task_outputs: TextCompletionOutputs = Field(default=Required)
