import os
import sys
import threading
from queue import Queue
from typing import Any, Optional

import torch
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.llms import (
    HuggingFaceTextGenInference,
    CTransformers,
    GPT4All,
    HuggingFacePipeline,
    LlamaCpp,
    OpenAI,
)
from langchain.schema import LLMResult
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    T5Tokenizer,
    TextStreamer,
    pipeline,
)

from app_modules.utils import ensure_model_is_downloaded


class TextIteratorStreamer(TextStreamer, StreamingStdOutCallbackHandler):
    def __init__(
        self,
        tokenizer: "AutoTokenizer",
        skip_prompt: bool = False,
        timeout: Optional[float] = None,
        for_huggingface: bool = False,
        **decode_kwargs,
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.text_queue = Queue()
        self.stop_signal = None
        self.timeout = timeout
        self.total_tokens = 0
        self.for_huggingface = for_huggingface

    def on_finalized_text(self, text: str, stream_end: bool = False):
        super().on_finalized_text(text, stream_end=stream_end)

        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
        self.text_queue.put(text, timeout=self.timeout)
        self.total_tokens = self.total_tokens + 1
        if stream_end:
            print("\n")
            self.text_queue.put("\n", timeout=self.timeout)
            self.text_queue.put(self.stop_signal, timeout=self.timeout)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        sys.stdout.write(token)
        sys.stdout.flush()
        self.text_queue.put(token, timeout=self.timeout)
        self.total_tokens = self.total_tokens + 1

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        print("\n")
        self.text_queue.put("\n", timeout=self.timeout)
        self.text_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value

    def reset(self, q: Queue = None):
        # print("resetting TextIteratorStreamer")
        self.text_queue = q if q is not None else Queue()

    def empty(self):
        return self.text_queue.empty()


class LLMLoader:
    llm_model_type: str
    llm: any
    streamer: any
    max_tokens_limit: int
    lock: any

    def __init__(self, llm_model_type):
        self.llm_model_type = llm_model_type
        self.llm = None
        self.streamer = TextIteratorStreamer(
            "",
            for_huggingface=True,
        )
        self.max_tokens_limit = 2048
        self.lock = threading.Lock()

    def _init_hf_streamer(self, tokenizer):
        self.streamer = TextIteratorStreamer(
            tokenizer,
            timeout=10.0,
            skip_prompt=True,
            skip_special_tokens=True,
            for_huggingface=True,
        )

    def init(
        self,
        custom_handler: Optional[BaseCallbackHandler] = None,
        n_threds: int = 4,
        hf_pipeline_device_type: str = None,
    ):
        print("initializing LLM: " + self.llm_model_type)

        if hf_pipeline_device_type is None:
            hf_pipeline_device_type = "cpu"

        using_cuda = hf_pipeline_device_type.startswith("cuda")
        torch_dtype = torch.float16 if using_cuda else torch.float32
        if os.environ.get("USING_TORCH_BFLOAT16") == "true":
            torch_dtype = torch.bfloat16
        load_quantized_model = os.environ.get("LOAD_QUANTIZED_MODEL")

        print(f"  hf_pipeline_device_type: {hf_pipeline_device_type}")
        print(f"     load_quantized_model: {load_quantized_model}")
        print(f"              torch_dtype: {torch_dtype}")
        print(f"                 n_threds: {n_threds}")

        double_quant_config = BitsAndBytesConfig(
            load_in_4bit=load_quantized_model == "4bit",
            bnb_4bit_use_double_quant=load_quantized_model == "4bit",
            load_in_8bit=load_quantized_model == "8bit",
            bnb_8bit_use_double_quant=load_quantized_model == "8bit",
        )

        callbacks = []
        if self.streamer is not None:
            callbacks.append(self.streamer)
        if custom_handler is not None:
            callbacks.append(custom_handler)

        if self.llm is None:
            if self.llm_model_type == "openai":
                MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME") or "gpt-3.5-turbo"
                print(f"              using model: {MODEL_NAME}")
                self.llm = (
                    OpenAI(
                        model_name=MODEL_NAME,
                        streaming=True,
                        callbacks=callbacks,
                        verbose=True,
                        temperature=0,
                    )
                    if "instruct" in MODEL_NAME
                    else ChatOpenAI(
                        model_name=MODEL_NAME,
                        streaming=True,
                        callbacks=callbacks,
                        verbose=True,
                        temperature=0,
                    )
                )
            elif self.llm_model_type.startswith("gpt4all"):
                MODEL_PATH = ensure_model_is_downloaded(self.llm_model_type)
                self.llm = GPT4All(
                    model=MODEL_PATH,
                    max_tokens=2048,
                    n_threads=n_threds,
                    backend="gptj" if self.llm_model_type == "gpt4all-j" else "llama",
                    callbacks=callbacks,
                    verbose=True,
                    use_mlock=True,
                )
            elif self.llm_model_type == "llamacpp":
                MODEL_PATH = ensure_model_is_downloaded(self.llm_model_type)
                self.llm = LlamaCpp(
                    model_path=MODEL_PATH,
                    n_ctx=8192,
                    n_threads=n_threds,
                    seed=0,
                    temperature=0,
                    max_tokens=2048,
                    callbacks=callbacks,
                    verbose=True,
                    use_mlock=True,
                )
            elif self.llm_model_type == "ctransformers":
                MODEL_PATH = ensure_model_is_downloaded(self.llm_model_type)
                config = {
                    "max_new_tokens": self.max_tokens_limit,
                    "repetition_penalty": 1.1,
                }
                self.llm = CTransformers(
                    model=MODEL_PATH,
                    model_type="llama",
                    config=config,
                    callbacks=callbacks,
                )
            elif self.llm_model_type == "hftgi":
                HFTGI_SERVER_URL = os.environ.get("HFTGI_SERVER_URL")
                HFTGI_RP = os.environ.get("HFTGI_RP")
                repetition_penalty = 1.120 if HFTGI_RP is None else float(HFTGI_RP)
                print(f"       repetition_penalty: {repetition_penalty}")
                self.max_tokens_limit = 4096
                self.llm = HuggingFaceTextGenInference(
                    inference_server_url=HFTGI_SERVER_URL,
                    max_new_tokens=self.max_tokens_limit / 2,
                    # top_k=0,
                    top_p=0.95,
                    # typical_p=0.95,
                    temperature=0.01,
                    repetition_penalty=repetition_penalty,
                    callbacks=callbacks,
                    timeout=600,
                    streaming=True,
                )
            elif self.llm_model_type.startswith("huggingface"):
                MODEL_NAME_OR_PATH = os.environ.get("HUGGINGFACE_MODEL_NAME_OR_PATH")
                print(f"            loading model: {MODEL_NAME_OR_PATH}")

                hf_auth_token = (
                    os.environ.get("HUGGINGFACE_AUTH_TOKEN")
                    if "Llama-2" in MODEL_NAME_OR_PATH
                    else None
                )
                transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE") == "1"
                token = (
                    hf_auth_token
                    if hf_auth_token is not None
                    and len(hf_auth_token) > 0
                    and not transformers_offline
                    else None
                )
                print(f"            HF auth token: {str(token)[-5:]}")

                if "Llama-2" in MODEL_NAME_OR_PATH:
                    self.max_tokens_limit = 4096

                is_t5 = "t5" in MODEL_NAME_OR_PATH
                temperature = (
                    0.01
                    if "gpt4all-j" in MODEL_NAME_OR_PATH
                    or "dolly" in MODEL_NAME_OR_PATH
                    or "Qwen" in MODEL_NAME_OR_PATH
                    or "Llama-2" in MODEL_NAME_OR_PATH
                    else 0
                )
                use_fast = (
                    "stable" in MODEL_NAME_OR_PATH
                    or "RedPajama" in MODEL_NAME_OR_PATH
                    or "dolly" in MODEL_NAME_OR_PATH
                )
                padding_side = "left"  # if "dolly" in MODEL_NAME_OR_PATH else None

                config = (
                    AutoConfig.from_pretrained(
                        MODEL_NAME_OR_PATH,
                        trust_remote_code=True,
                        token=token,
                        fp32=hf_pipeline_device_type == "cpu",
                        bf16=(
                            hf_pipeline_device_type != "cpu"
                            and torch_dtype == torch.bfloat16
                        ),
                        fp16=(
                            hf_pipeline_device_type != "cpu"
                            and torch_dtype != torch.bfloat16
                        ),
                    )
                    if "Qwen" in MODEL_NAME_OR_PATH
                    else AutoConfig.from_pretrained(
                        MODEL_NAME_OR_PATH,
                        trust_remote_code=True,
                        token=token,
                    )
                )

                # config.attn_config["attn_impl"] = "triton"
                # config.max_seq_len = 4096
                config.init_device = hf_pipeline_device_type

                tokenizer = (
                    T5Tokenizer.from_pretrained(
                        MODEL_NAME_OR_PATH,
                        token=token,
                    )
                    if is_t5
                    else AutoTokenizer.from_pretrained(
                        MODEL_NAME_OR_PATH,
                        use_fast=use_fast,
                        trust_remote_code=True,
                        padding_side=padding_side,
                        token=token,
                    )
                )

                self._init_hf_streamer(tokenizer)

                task = "text2text-generation" if is_t5 else "text-generation"

                return_full_text = True if "dolly" in MODEL_NAME_OR_PATH else None

                repetition_penalty = (
                    1.15
                    if "falcon" in MODEL_NAME_OR_PATH
                    else (1.25 if "dolly" in MODEL_NAME_OR_PATH else 1.1)
                )

                HF_RP = os.environ.get("HF_RP")
                if HF_RP is not None and len(HF_RP) > 0:
                    repetition_penalty = float(HF_RP)
                print(f"       repetition_penalty: {repetition_penalty}")

                if load_quantized_model is not None:
                    model = (
                        AutoModelForSeq2SeqLM.from_pretrained(
                            MODEL_NAME_OR_PATH,
                            config=config,
                            quantization_config=double_quant_config,
                            trust_remote_code=True,
                            use_auth_token=token,
                        )
                        if is_t5
                        else AutoModelForCausalLM.from_pretrained(
                            MODEL_NAME_OR_PATH,
                            config=config,
                            quantization_config=double_quant_config,
                            trust_remote_code=True,
                            use_auth_token=token,
                        )
                    )

                    print(f"Model memory footprint: {model.get_memory_footprint()}")

                    eos_token_id = -1
                    # starchat-beta uses a special <|end|> token with ID 49155 to denote ends of a turn
                    if "starchat" in MODEL_NAME_OR_PATH:
                        eos_token_id = 49155
                    pad_token_id = eos_token_id

                    pipe = (
                        InstructionTextGenerationPipeline(
                            task=task,
                            model=model,
                            tokenizer=tokenizer,
                            streamer=self.streamer,
                            max_new_tokens=2048,
                            temperature=temperature,
                            return_full_text=return_full_text,  # langchain expects the full text
                            repetition_penalty=repetition_penalty,
                        )
                        if "dolly" in MODEL_NAME_OR_PATH
                        else (
                            pipeline(
                                task,
                                model=model,
                                tokenizer=tokenizer,
                                eos_token_id=eos_token_id,
                                pad_token_id=pad_token_id,
                                streamer=self.streamer,
                                return_full_text=return_full_text,  # langchain expects the full text
                                device_map="auto",
                                trust_remote_code=True,
                                max_new_tokens=2048,
                                do_sample=True,
                                temperature=0.01,
                                top_p=0.95,
                                top_k=50,
                                repetition_penalty=repetition_penalty,
                            )
                            if eos_token_id != -1
                            else pipeline(
                                task,
                                model=model,
                                tokenizer=tokenizer,
                                streamer=self.streamer,
                                return_full_text=return_full_text,  # langchain expects the full text
                                device_map="auto",
                                trust_remote_code=True,
                                max_new_tokens=2048,
                                # verbose=True,
                                temperature=temperature,
                                top_p=0.95,
                                top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
                                repetition_penalty=repetition_penalty,
                            )
                        )
                    )
                else:
                    if os.environ.get("DISABLE_MODEL_PRELOADING") != "true":
                        model = (
                            AutoModelForSeq2SeqLM.from_pretrained(
                                MODEL_NAME_OR_PATH,
                                config=config,
                                trust_remote_code=True,
                            )
                            if is_t5
                            else (
                                AutoModelForCausalLM.from_pretrained(
                                    MODEL_NAME_OR_PATH,
                                    config=config,
                                    trust_remote_code=True,
                                )
                                if "Qwen" in MODEL_NAME_OR_PATH
                                else (
                                    AutoModelForCausalLM.from_pretrained(
                                        MODEL_NAME_OR_PATH,
                                        config=config,
                                        trust_remote_code=True,
                                    )
                                    if token is None
                                    else AutoModelForCausalLM.from_pretrained(
                                        MODEL_NAME_OR_PATH,
                                        config=config,
                                        trust_remote_code=True,
                                        token=token,
                                    )
                                )
                            )
                        )
                        print(f"Model memory footprint: {model.get_memory_footprint()}")
                        model = model.eval()
                        # print(f"Model memory footprint: {model.get_memory_footprint()}")
                    else:
                        model = MODEL_NAME_OR_PATH

                    pipe = (
                        pipeline(
                            task,
                            model=model,
                            tokenizer=tokenizer,
                            streamer=self.streamer,
                            return_full_text=return_full_text,  # langchain expects the full text
                            device=hf_pipeline_device_type,
                            torch_dtype=torch_dtype,
                            max_new_tokens=2048,
                            trust_remote_code=True,
                            temperature=temperature,
                            top_p=0.95,
                            top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
                            repetition_penalty=repetition_penalty,
                        )
                        if token is None
                        else pipeline(
                            task,
                            model=model,
                            tokenizer=tokenizer,
                            streamer=self.streamer,
                            return_full_text=return_full_text,  # langchain expects the full text
                            device=hf_pipeline_device_type,
                            torch_dtype=torch_dtype,
                            max_new_tokens=2048,
                            trust_remote_code=True,
                            temperature=temperature,
                            top_p=0.95,
                            top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
                            repetition_penalty=repetition_penalty,
                            token=token,
                        )
                    )

                self.llm = HuggingFacePipeline(pipeline=pipe, callbacks=callbacks)
            elif self.llm_model_type == "mosaicml":
                MODEL_NAME_OR_PATH = os.environ.get("MOSAICML_MODEL_NAME_OR_PATH")
                print(f"            loading model: {MODEL_NAME_OR_PATH}")

                config = AutoConfig.from_pretrained(
                    MODEL_NAME_OR_PATH, trust_remote_code=True
                )
                # config.attn_config["attn_impl"] = "triton"
                config.max_seq_len = 16384 if "30b" in MODEL_NAME_OR_PATH else 4096
                config.init_device = hf_pipeline_device_type

                model = (
                    AutoModelForCausalLM.from_pretrained(
                        MODEL_NAME_OR_PATH,
                        config=config,
                        quantization_config=double_quant_config,
                        trust_remote_code=True,
                    )
                    if load_quantized_model is not None
                    else AutoModelForCausalLM.from_pretrained(
                        MODEL_NAME_OR_PATH,
                        config=config,
                        torch_dtype=torch_dtype,
                        trust_remote_code=True,
                    )
                )

                print(f"Model loaded on {config.init_device}")
                print(f"Model memory footprint: {model.get_memory_footprint()}")

                tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
                self._init_hf_streamer(tokenizer)

                # mtp-7b is trained to add "<|endoftext|>" at the end of generations
                stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])

                # define custom stopping criteria object
                class StopOnTokens(StoppingCriteria):
                    def __call__(
                        self,
                        input_ids: torch.LongTensor,
                        scores: torch.FloatTensor,
                        **kwargs,
                    ) -> bool:
                        for stop_id in stop_token_ids:
                            if input_ids[0][-1] == stop_id:
                                return True
                        return False

                stopping_criteria = StoppingCriteriaList([StopOnTokens()])

                max_new_tokens = 8192 if "30b" in MODEL_NAME_OR_PATH else 2048
                self.max_tokens_limit = max_new_tokens
                repetition_penalty = 1.05 if "30b" in MODEL_NAME_OR_PATH else 1.02

                ML_RP = os.environ.get("ML_RP")
                if ML_RP is not None and len(ML_RP) > 0:
                    repetition_penalty = float(ML_RP)
                print(f"       repetition_penalty: {repetition_penalty}")

                pipe = (
                    pipeline(
                        model=model,
                        tokenizer=tokenizer,
                        streamer=self.streamer,
                        return_full_text=True,  # langchain expects the full text
                        task="text-generation",
                        device_map="auto",
                        # we pass model parameters here too
                        stopping_criteria=stopping_criteria,  # without this model will ramble
                        do_sample=True,
                        temperature=0.01,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
                        top_p=0.95,  # select from top tokens whose probability add up to 15%
                        top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
                        max_new_tokens=max_new_tokens,  # mex number of tokens to generate in the output
                        repetition_penalty=repetition_penalty,  # without this output begins repeating
                    )
                    if load_quantized_model is not None
                    else pipeline(
                        model=model,
                        tokenizer=tokenizer,
                        streamer=self.streamer,
                        return_full_text=True,  # langchain expects the full text
                        task="text-generation",
                        device=config.init_device,
                        # we pass model parameters here too
                        stopping_criteria=stopping_criteria,  # without this model will ramble
                        do_sample=True,
                        temperature=0.01,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
                        top_p=0.95,  # select from top tokens whose probability add up to 15%
                        top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
                        max_new_tokens=max_new_tokens,  # mex number of tokens to generate in the output
                        repetition_penalty=repetition_penalty,  # without this output begins repeating
                    )
                )
                self.llm = HuggingFacePipeline(pipeline=pipe, callbacks=callbacks)
            elif self.llm_model_type == "stablelm":
                MODEL_NAME_OR_PATH = os.environ.get("STABLELM_MODEL_NAME_OR_PATH")
                print(f"            loading model: {MODEL_NAME_OR_PATH}")

                config = AutoConfig.from_pretrained(
                    MODEL_NAME_OR_PATH, trust_remote_code=True
                )
                # config.attn_config["attn_impl"] = "triton"
                # config.max_seq_len = 4096
                config.init_device = hf_pipeline_device_type

                SL_RP = os.environ.get("SL_RP")
                if SL_RP is not None and len(SL_RP) > 0:
                    repetition_penalty = float(SL_RP)
                else:
                    repetition_penalty = 1.05
                print(f"       repetition_penalty: {repetition_penalty}")

                model = (
                    AutoModelForCausalLM.from_pretrained(
                        MODEL_NAME_OR_PATH,
                        config=config,
                        quantization_config=double_quant_config,
                        trust_remote_code=True,
                    )
                    if load_quantized_model is not None
                    else AutoModelForCausalLM.from_pretrained(
                        MODEL_NAME_OR_PATH,
                        config=config,
                        torch_dtype=torch_dtype,
                        trust_remote_code=True,
                    )
                )

                print(f"Model loaded on {config.init_device}")
                print(f"Model memory footprint: {model.get_memory_footprint()}")

                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
                self._init_hf_streamer(tokenizer)

                class StopOnTokens(StoppingCriteria):
                    def __call__(
                        self,
                        input_ids: torch.LongTensor,
                        scores: torch.FloatTensor,
                        **kwargs,
                    ) -> bool:
                        stop_ids = [50278, 50279, 50277, 1, 0]
                        for stop_id in stop_ids:
                            if input_ids[0][-1] == stop_id:
                                return True
                        return False

                stopping_criteria = StoppingCriteriaList([StopOnTokens()])

                pipe = (
                    pipeline(
                        model=model,
                        tokenizer=tokenizer,
                        streamer=self.streamer,
                        return_full_text=True,  # langchain expects the full text
                        task="text-generation",
                        device_map="auto",
                        # we pass model parameters here too
                        stopping_criteria=stopping_criteria,  # without this model will ramble
                        temperature=0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
                        top_p=0.95,  # select from top tokens whose probability add up to 15%
                        top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
                        max_new_tokens=2048,  # mex number of tokens to generate in the output
                        repetition_penalty=repetition_penalty,  # without this output begins repeating
                    )
                    if load_quantized_model is not None
                    else pipeline(
                        model=model,
                        tokenizer=tokenizer,
                        streamer=self.streamer,
                        return_full_text=True,  # langchain expects the full text
                        task="text-generation",
                        device=config.init_device,
                        # we pass model parameters here too
                        stopping_criteria=stopping_criteria,  # without this model will ramble
                        temperature=0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
                        top_p=0.95,  # select from top tokens whose probability add up to 15%
                        top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
                        max_new_tokens=2048,  # mex number of tokens to generate in the output
                        repetition_penalty=repetition_penalty,  # without this output begins repeating
                    )
                )
                self.llm = HuggingFacePipeline(pipeline=pipe, callbacks=callbacks)

        print("initialization complete")
