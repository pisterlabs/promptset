from src import openai_utils
import datetime
from abc import ABC, abstractmethod
from asyncio import Queue, get_event_loop, wait_for, TimeoutError, Future
from transformers import (
    pipeline,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch


OAI_CHAT_MODELS = {"gpt-3.5-turbo-0613", "gpt-4-0613","gpt-3.5-turbo"}
OAI_LEGACY_MODELS = {"text-davinci-003"}
HF_LLAMA_CHAT_MODELS = {
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "meta-llama/Llama-2-7b-chat-hf-4bit",
    "meta-llama/Llama-2-13b-chat-hf-4bit",
    "meta-llama/Llama-2-70b-chat-hf-4bit",
}
HF_MODELS = {
    "gpt2",
    "gpt2-xl",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-hf",
}



HF_GENERATOR_CACHE = {}


def get_cached_hf_generator(model_name):
    """
    Only want to load each model once.
    Return a cached model instance if it exists; otherwise, create, cache, and return a new instance.
    """

    if model_name not in HF_GENERATOR_CACHE:
        if model_name in HF_MODELS | HF_LLAMA_CHAT_MODELS:
            if model_name.endswith("-4bit"):
                model_name = model_name[:-5]
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )

                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    quantization_config=bnb_config,
                    device_map="auto",
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    device_map="auto",
                )

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Set pad to eos if there is no pad
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
                
            HF_GENERATOR_CACHE[model_name] = pipeline(
                "text-generation",
                # return_full_text=True,
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    return HF_GENERATOR_CACHE[model_name]


def structure_message(role: str, content: str) -> dict:
    if role not in {"system", "user", "assistant"}:
        raise ValueError(f"Unknown chat role: {role}")
    return {"role": role, "content": content}


def get_model(model_name: str, **kwargs):
    if model_name in OAI_CHAT_MODELS | OAI_LEGACY_MODELS:
        return GPTModelInstance(model_name=model_name, **kwargs)
    elif model_name in HF_LLAMA_CHAT_MODELS:
        return HfLlamaChatModelInstance(model_name=model_name, **kwargs)
    elif model_name in HF_MODELS:
        return HfModelInstance(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


# Create an ABC for Models
class ModelInstance(ABC):
    def __init__(
        self,
        model_name: str,
        system_message="",
        prompt="",
        pre_prompt="",
        temperature=0.7,
        max_tokens=256
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens

        if (
            model_name
            not in OAI_CHAT_MODELS
            | OAI_LEGACY_MODELS
            | HF_MODELS
            | HF_LLAMA_CHAT_MODELS
        ):
            raise ValueError(f"Unknown model name: {model_name}")

        self.model_name = model_name
        self.is_chat_model = model_name in OAI_CHAT_MODELS

    @abstractmethod
    async def generate_async(self, content: str, n_sample: int):
        pass


class GPTModelInstance(ModelInstance):
    def __init__(
        self,
        model_name: str,
        system_message="",
        prompt="",
        pre_prompt="",
        temperature=0.7,
        max_tokens=256
    ):
        super().__init__(model_name=model_name,
                         system_message=system_message,
                         prompt=prompt,
                         pre_prompt=pre_prompt,
                         temperature=temperature,
                         max_tokens=max_tokens)

        self.system_message = structure_message(
            "system", system_message
        )  if system_message != "" else None # System message to prepend to all queries
        self.prompt_message = structure_message(
            "user", prompt
        )  if prompt != "" else None  # Prompt message to append to all queries
        self.pre_prompt_message = structure_message(
            "user", pre_prompt
        )  if pre_prompt != "" else None  # Prompt message to append to all queries

        self.tokenizer = openai_utils.get_tokenizer(model_name)

    async def generate_async(
        self, content: str, n_sample: int = 1, logit_bias: dict = {}
    ):
        query_messages = [
            self.system_message,
            self.pre_prompt_message,
            structure_message("user", content),
            self.prompt_message,
        ]
        query_messages = [m for m in query_messages if m is not None]

        if self.is_chat_model:
            response = await openai_utils.chat_with_backoff_async(
                model=self.model_name,
                messages=query_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                logit_bias=logit_bias,
                n=n_sample,
            )

        samples = [s["message"]["content"] for s in response["choices"]]
        return {
                "input": query_messages,
                "response": samples,
                "token_counts": response["usage"]
        }

    def prompts_cost(self):
        return openai_utils.cost(self.prompt_tokens, self.model_name, "prompt")

    def completions_cost(self):
        return openai_utils.cost(self.completion_tokens, self.model_name, "completion")

    def total_cost(self):
        return self.prompts_cost() + self.completions_cost()


class HfModelInstance(ModelInstance):
    def __init__(
        self,
        model_name: str,
        system_message="",
        prompt="",
        pre_prompt="",
        temperature=0.7,
        max_tokens=512,
        batch_size=1
    ):
        super().__init__(model_name, system_message, prompt, temperature, max_tokens)

        self.system_message = system_message
        self.prompt_message = prompt
        self.pre_prompt_message = pre_prompt

        self.batch_size = batch_size
        
        self.generator = get_cached_hf_generator(model_name)
        
        self.async_queue = Queue()
        self.loop = get_event_loop()
        self.loop.create_task(self.process_batches())
        
    def format_input(self, system_message: str, content: str, prompt_message: str):
        return "\n\n".join([content, prompt_message])
    
    async def add_request(self, content: str, n_sample=1):
        future = Future()
        await self.async_queue.put((content, n_sample, future))
        return future

    async def process_batches(self):
        # Repeatedly fill the batch until batch_size is reached or timeout, then process the batch
        while True:
            batch = []
            for _ in range(self.batch_size):
                try:
                    item = await wait_for(self.async_queue.get(), timeout=30.0)
                    batch.append(item)
                except TimeoutError:
                    if batch:
                        break
            await self.process_batch(batch)

    async def process_batch(self, batch):
        # Process a batch of requests concurrently
        input_texts = [
            self.format_input(self.system_message, content, self.prompt_message)
            for content, _, _ in batch
        ]

        print(f"Processing batch of size {len(batch)} (max: {self.batch_size})")
        if not input_texts: # Skip if the batch is empty
            return
        start_time = datetime.datetime.now()
        batch_output = self.generator(
            input_texts,
            num_return_sequences=1,
            return_tensors=True,
            do_sample=True,
            temperature=self.temperature,
            max_length=self.max_tokens,
            repetition_penalty=1.1
        )
        print(f"Took {datetime.datetime.now() - start_time} to generate {len(batch)} samples")

        # Iterate through the responses for each request in the batch
        for i, completion_tokens in enumerate(batch_output):
            completion_tokens = completion_tokens[0]["generated_token_ids"] # Extract the generated token ids
            future = batch[i][2]  # This request's future
            generated_text = self.generator.tokenizer(completion_tokens, skip_special_tokens=True)

            # Count input and output tokens
            prompt_tokens = self.generator.tokenizer.encode(input_texts[i], return_tensors="pt")
            n_prompt_tokens = len(prompt_tokens['input_ids'])
            n_completion_tokens = len(completion_tokens)

            # Set this request's future result
            future.set_result({
                "input": input_texts[i],
                "response": [generated_text],
                "token_counts": {
                        "prompt_tokens": n_prompt_tokens,
                        "completion_tokens": n_completion_tokens,
                        "total_tokens": n_prompt_tokens + n_completion_tokens
                    }
            })

    async def generate_async(self, content: str, n_sample: int = 1):
        future = await self.add_request(content, n_sample)
        return await future  # Await the future and return its result


class HfLlamaChatModelInstance(HfModelInstance):
    # The chat versions of the Llama2 models are fine tuned to use a specific prompt format.
    def format_input(self, system_message: str, content: str, prompt_message: str):
        prompt_part = f'\n\n{prompt_message}' if prompt_message != '' else ''
        return f"<s><<SYS>>\n{system_message}\n<</SYS>>\n\n[INST]{content}{prompt_part}[/INST] "
