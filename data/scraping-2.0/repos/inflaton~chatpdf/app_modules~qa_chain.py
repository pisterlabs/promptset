import os
import sys
import urllib
from queue import Queue
from typing import Any, Optional

import torch
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import GPT4All, HuggingFacePipeline, LlamaCpp
from langchain.schema import LLMResult
from langchain.vectorstores import VectorStore
from langchain.vectorstores.base import VectorStore
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

from app_modules.instruct_pipeline import InstructionTextGenerationPipeline
from app_modules.utils import ensure_model_is_downloaded, remove_extra_spaces


class TextIteratorStreamer(TextStreamer, StreamingStdOutCallbackHandler):
    def __init__(
        self,
        tokenizer: "AutoTokenizer",
        skip_prompt: bool = False,
        timeout: Optional[float] = None,
        **decode_kwargs,
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.text_queue = Queue()
        self.stop_signal = None
        self.timeout = timeout

    def on_finalized_text(self, text: str, stream_end: bool = False):
        super().on_finalized_text(text, stream_end=stream_end)

        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
        self.text_queue.put(text, timeout=self.timeout)
        if stream_end:
            print("\n")
            self.text_queue.put("\n", timeout=self.timeout)
            self.text_queue.put(self.stop_signal, timeout=self.timeout)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        sys.stdout.write(token)
        sys.stdout.flush()
        self.text_queue.put(token, timeout=self.timeout)

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


class QAChain:
    llm_model_type: str
    vectorstore: VectorStore
    llm: any
    streamer: any

    def __init__(self, vectorstore, llm_model_type):
        self.vectorstore = vectorstore
        self.llm_model_type = llm_model_type
        self.llm = None
        self.streamer = TextIteratorStreamer("")
        self.max_tokens_limit = 2048
        self.search_kwargs = {"k": 4}

    def _init_streamer(self, tokenizer, custom_handler):
        self.streamer = (
            TextIteratorStreamer(
                tokenizer,
                timeout=10.0,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            if custom_handler is None
            else TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
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

        callbacks = [self.streamer]
        if custom_handler is not None:
            callbacks.append(custom_handler)

        if self.llm is None:
            if self.llm_model_type == "openai":
                MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME") or "gpt-4"
                print(f"              using model: {MODEL_NAME}")
                self.llm = ChatOpenAI(
                    model_name=MODEL_NAME,
                    streaming=True,
                    callbacks=callbacks,
                    verbose=True,
                    temperature=0,
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
            elif self.llm_model_type.startswith("huggingface"):
                MODEL_NAME_OR_PATH = os.environ.get("HUGGINGFACE_MODEL_NAME_OR_PATH")
                print(f"            loading model: {MODEL_NAME_OR_PATH}")

                hf_auth_token = os.environ.get("HUGGINGFACE_AUTH_TOKEN")
                use_auth_token = (
                    hf_auth_token
                    if hf_auth_token is not None and len(hf_auth_token) > 0
                    else False
                )
                print(f"           use_auth_token: {str(use_auth_token)[-5:]}")

                is_t5 = "t5" in MODEL_NAME_OR_PATH
                temperature = (
                    0.01
                    if "gpt4all-j" in MODEL_NAME_OR_PATH
                    or "dolly" in MODEL_NAME_OR_PATH
                    else 0
                )
                use_fast = (
                    "stable" in MODEL_NAME_OR_PATH
                    or "RedPajama" in MODEL_NAME_OR_PATH
                    or "dolly" in MODEL_NAME_OR_PATH
                )
                padding_side = "left"  # if "dolly" in MODEL_NAME_OR_PATH else None

                config = AutoConfig.from_pretrained(
                    MODEL_NAME_OR_PATH,
                    trust_remote_code=True,
                    use_auth_token=use_auth_token,
                )
                # config.attn_config["attn_impl"] = "triton"
                # config.max_seq_len = 4096
                config.init_device = hf_pipeline_device_type

                tokenizer = (
                    T5Tokenizer.from_pretrained(
                        MODEL_NAME_OR_PATH,
                        use_auth_token=use_auth_token,
                    )
                    if is_t5
                    else AutoTokenizer.from_pretrained(
                        MODEL_NAME_OR_PATH,
                        use_fast=use_fast,
                        trust_remote_code=True,
                        padding_side=padding_side,
                        use_auth_token=use_auth_token,
                    )
                )

                self._init_streamer(tokenizer, custom_handler)

                task = "text2text-generation" if is_t5 else "text-generation"

                return_full_text = True if "dolly" in MODEL_NAME_OR_PATH else None

                repetition_penalty = (
                    1.15
                    if "falcon" in MODEL_NAME_OR_PATH
                    else (1.25 if "dolly" in MODEL_NAME_OR_PATH else 1.1)
                )

                if load_quantized_model is not None:
                    model = (
                        AutoModelForSeq2SeqLM.from_pretrained(
                            MODEL_NAME_OR_PATH,
                            config=config,
                            quantization_config=double_quant_config,
                            trust_remote_code=True,
                            use_auth_token=use_auth_token,
                        )
                        if is_t5
                        else AutoModelForCausalLM.from_pretrained(
                            MODEL_NAME_OR_PATH,
                            config=config,
                            quantization_config=double_quant_config,
                            trust_remote_code=True,
                            use_auth_token=use_auth_token,
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
                            use_auth_token=use_auth_token,
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
                                use_auth_token=use_auth_token,
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
                                use_auth_token=use_auth_token,
                            )
                        )
                    )
                elif "dolly" in MODEL_NAME_OR_PATH:
                    model = AutoModelForCausalLM.from_pretrained(
                        MODEL_NAME_OR_PATH,
                        device_map=hf_pipeline_device_type,
                        torch_dtype=torch_dtype,
                    )

                    pipe = InstructionTextGenerationPipeline(
                        task=task,
                        model=model,
                        tokenizer=tokenizer,
                        streamer=self.streamer,
                        max_new_tokens=2048,
                        temperature=temperature,
                        return_full_text=True,
                        repetition_penalty=repetition_penalty,
                        use_auth_token=use_auth_token,
                    )
                else:
                    pipe = pipeline(
                        task,  # model=model,
                        model=MODEL_NAME_OR_PATH,
                        tokenizer=tokenizer,
                        streamer=self.streamer,
                        return_full_text=return_full_text,  # langchain expects the full text
                        device=hf_pipeline_device_type,
                        torch_dtype=torch_dtype,
                        max_new_tokens=2048,
                        trust_remote_code=True,
                        # verbose=True,
                        temperature=temperature,
                        top_p=0.95,
                        top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
                        repetition_penalty=1.115,
                        use_auth_token=use_auth_token,
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
                self._init_streamer(tokenizer, custom_handler)

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
                self.search_kwargs = (
                    {"k": 8} if "30b" in MODEL_NAME_OR_PATH else self.search_kwargs
                )
                repetition_penalty = 1.05 if "30b" in MODEL_NAME_OR_PATH else 1.02

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
                        temperature=0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
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
                self._init_streamer(tokenizer, custom_handler)

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
                        repetition_penalty=1.25,  # without this output begins repeating
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
                        repetition_penalty=1.05,  # without this output begins repeating
                    )
                )
                self.llm = HuggingFacePipeline(pipeline=pipe, callbacks=callbacks)

        print("initialization complete")

    def get_chain(self, tracing: bool = False) -> ConversationalRetrievalChain:
        if tracing:
            tracer = LangChainTracer()
            tracer.load_default_session()

        if self.llm is None:
            self.init()

        qa = ConversationalRetrievalChain.from_llm(
            self.llm,
            self.vectorstore.as_retriever(search_kwargs=self.search_kwargs),
            max_tokens_limit=self.max_tokens_limit,
            return_source_documents=True,
        )

        return qa

    def call(self, inputs, q: Queue = None, tracing: bool = False):
        print(inputs)

        if self.streamer is not None and isinstance(
            self.streamer, TextIteratorStreamer
        ):
            self.streamer.reset(q)

        qa = self.get_chain(tracing)
        result = qa(inputs)

        result["answer"] = remove_extra_spaces(result["answer"])

        base_url = os.environ.get("PDF_FILE_BASE_URL")
        if base_url is not None and len(base_url) > 0:
            documents = result["source_documents"]
            for doc in documents:
                source = doc.metadata["source"]
                title = source.split("/")[-1]
                doc.metadata["url"] = f"{base_url}{urllib.parse.quote(title)}"

        return result
