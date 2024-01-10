import logging
from typing import Any, Dict, List, Optional, Tuple
import os
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    TextStreamer,
)

from democratic_agent.chat.conversation import Conversation

# Huggingface considerations:
# AutoModelForCausalLM already fetches from source code to check for the base class or in "Architectures" at config.json from model card if no trust_remote_code is set. So we can make it general to run all models based on common transformer interface.
# torch_dtype can be set manually or using auto. Auto will look into config.json or fetch the `dtype` from the first weight in the checkpoint
# Huggingface requires a conversation: Union[List[Dict[str, str]], "Conversation"] to apply chat template. This is why I set prompt to Any, as can str, dict, Conversation... Lets figure out in the future for right hint.


LOG = logging.getLogger(__name__)


class OSModel:
    def __init__(self, model_name: str, model_revision: Optional[str] = None):
        self.model, self.tokenizer = self._load_or_download_model(
            model_name, model_revision
        )
        self.name = model_name
        # TODO: get debug from cfg
        debug = False
        if debug:
            self.streamer = TextStreamer(self.tokenizer, skip_prompt=True)
        else:
            self.streamer = None

    def _load_or_download_model(
        self, model_name: str, model_revision: Optional[str] = None, device="cuda"
    ):
        local_model_name = model_name.replace("/", "_")
        local_model_path = os.path.join(
            Path(__file__).parent, "local_models", local_model_name
        )

        def _load_model(model_path) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                revision=model_revision,
                device_map=device,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            return model, tokenizer

        if os.path.exists(local_model_path):
            print(f"Loading model from local path: {local_model_path}")
            model, tokenizer = _load_model(local_model_path)
        else:
            print(
                f"Model not found locally. Downloading from Hugging Face: {model_name}"
            )
            model, tokenizer = _load_model(model_name)

            # Save the model locally for future use
            model.save_pretrained(local_model_path)
            tokenizer.save_pretrained(local_model_path)

        if tokenizer.chat_template:
            LOG.info(f"The model as a chat template: {tokenizer.chat_template}")
        return model, tokenizer

    # TODO: IMPLEMENT ME -> Check OpenAI imp -> Add conversation and functions, we should translate it to ChatCompletionMessageToolCall from OpenAI to use a common interface.
    def get_response(
        self,
        conversation: Conversation,
        functions: List[Dict[str, Any]] = [],
        response_format: str = "text",  # or json_object.
        temperature: float = 0.7,
    ):
        # TODO: If not chat_template we can create it manually?
        if self.tokenizer.chat_template:
            input_ids = self.tokenizer.apply_chat_template(
                conversation, return_tensors="pt", add_generation_prompt=True
            ).to(self.model.device)
            tokens = self.model.generate(
                input_ids=input_ids,
                do_sample=True,
                temperature=1.0,
                max_length=2048,  # TODO: Get from huggingface or local config (ideally automatic...)
                streamer=self.streamer,
            )
        # TODO: Merge this into a single call.
        else:
            input_ids = (
                self.tokenizer(prompt, return_tensors="pt")
                .to(self.model.device)
                .input_ids
            )  # Why do I get a hint with prompt as str?
            tokens = self.model.generate(
                input_ids=input_ids,
                temperature=1.0,
                max_length=2048,  # TODO: Get from huggingface or local config (ideally automatic...)
                streamer=self.streamer,
            )

        response = self.tokenizer.batch_decode(
            tokens[:, input_ids.shape[1] :], skip_special_tokens=True
        )[0]
        return response
