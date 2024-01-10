from pathlib import Path

from cot_transparency.data_models.config import OpenaiInferenceConfig
from cot_transparency.data_models.messages import ChatMessage, MessageRole
from cot_transparency.data_models.models import ModelOutput, TaskOutput, TaskSpec
from cot_transparency.formatters.core.sycophancy import ZeroShotCOTSycophancyFormatter
from cot_transparency.util import deterministic_hash


def test_taskspec_and_taskoutput_give_same():
    messages = []
    messages.append(ChatMessage(role=MessageRole.user, content="This is the first messages"))

    config = OpenaiInferenceConfig(model="gpt-4", max_tokens=10, temperature=0.5, top_p=1.0)

    task_hash = deterministic_hash("dummy input for hash function")
    task_spec = TaskSpec(
        task_name="task_name",
        inference_config=config,
        messages=messages,
        out_file_path=Path("out_file_path"),
        ground_truth="A",
        formatter_name=ZeroShotCOTSycophancyFormatter.name(),
        task_hash=task_hash,
        data_example={},
    )

    dummy_output = ModelOutput(raw_response="The best answer is: (A)", parsed_response="A")
    task_output = TaskOutput(
        task_spec=task_spec,
        inference_output=dummy_output,
    )

    assert task_output.task_spec_uid() == task_spec.uid()
