"""This module will be spawned in process pool, so parent process can't its global variables."""
from typing import TYPE_CHECKING, Generator

from langchain import LlamaCpp
from llama_cpp import Llama

from app.exceptions import (
    GptBreakException,
    GptContinueException,
    GptLengthException,
    GptTextGenerationException,
)
from database.dataclasses import ChatGPTConfig

if TYPE_CHECKING:
    from gpt.common import LlamaCppModel, UserGptContext


def can_avoid_in_buffer(text_buffer: str, text: str, avoids: list[str]) -> bool:
    for avoid in avoids:
        avoid = avoid.upper()
        possible_buffer = (text_buffer + text).upper()
        if avoid in possible_buffer or any(
            [possible_buffer.endswith(avoid[: i + 1]) for i in range(len(avoid))]
        ):
            return True
    return False


def get_stops(s: str) -> list[str]:
    return [
        s,
        s.upper(),
        s.lower(),
        s.capitalize(),
    ]


def get_fake_response() -> Generator:
    from random import randint

    yield {
        "id": "fake_id",
        "object": "text_completion",
        "created": 12345,
        "model": "./fake_models/ggml/fake-llama-13B.ggml.q4_2.bin",
        "choices": [
            {
                "text": str(randint(0, 9)),
                "index": 0,
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    }


def load_llama(llama_cpp_model: "LlamaCppModel") -> LlamaCpp:
    model_name: str = llama_cpp_model.name
    if model_name not in globals():
        globals()[model_name] = get_llama(llama_cpp_model)
    return globals()[model_name]


def llama_cpp_generation(
    llama_cpp_model: "LlamaCppModel",
    prompt: str,
    m_queue,  # multiprocessing.managers.AutoProxy[Queue]
    m_done,  # multiprocessing.managers.EventProxy
    user_gpt_context: "UserGptContext",
    is_fake: bool = False,
    use_client_only: bool = True,
) -> None:
    m_done.clear()
    blank: str = "\u200b"
    avoids: list[str] = get_stops(
        user_gpt_context.user_gpt_profile.user_role + ":",
    ) + get_stops(user_gpt_context.user_gpt_profile.gpt_role + ":")
    llm = load_llama(llama_cpp_model)
    llm_client: Llama = llm.client
    llm_client.verbose = bool(llm.echo)
    retry_count: int = 0

    while True:
        retry_count += 1
        if use_client_only:
            generator = llm_client.create_completion(  # type: ignore
                prompt=prompt,
                suffix=llm.suffix,
                max_tokens=min(
                    user_gpt_context.left_tokens,
                    user_gpt_context.gpt_model.value.max_tokens_per_request,
                ),
                temperature=user_gpt_context.user_gpt_profile.temperature,
                top_p=user_gpt_context.user_gpt_profile.top_p,
                logprobs=llm.logprobs,
                echo=bool(llm.echo),
                stop=llm.stop + avoids if llm.stop is not None else avoids,
                repeat_penalty=user_gpt_context.user_gpt_profile.frequency_penalty,
                top_k=40,
                stream=True,
            )

        else:
            llm.temperature = user_gpt_context.user_gpt_profile.temperature
            llm.top_p = user_gpt_context.user_gpt_profile.top_p
            llm.max_tokens = min(
                user_gpt_context.left_tokens,
                user_gpt_context.gpt_model.value.max_tokens_per_request,
            )
            generator = (
                llm.stream(
                    prompt=prompt,
                    stop=llm.stop + avoids if llm.stop is not None else avoids,
                )
                if not is_fake
                else get_fake_response()
            )

        content_buffer: str = ""
        deleted_histories: int = 0

        try:
            for generation in generator:
                if m_done.is_set() or retry_count > 10:
                    m_queue.put_nowait(
                        GptTextGenerationException(msg="Max retry count reached")
                    )
                    m_done.set()
                    return  # stop generating if main process requests to stop
                finish_reason: str | None = generation["choices"][0]["finish_reason"]  # type: ignore
                text: str = generation["choices"][0]["text"]  # type: ignore
                if text.replace(blank, "") == "":
                    continue
                if content_buffer == "":
                    text = text.lstrip()
                    if text == "":
                        continue
                if finish_reason == "length":
                    raise GptLengthException(
                        msg="Incomplete model output due to max_tokens parameter or token limit"
                    )  # raise exception for token limit
                content_buffer += text
                m_queue.put(text)
            if content_buffer.replace(blank, "").strip() == "":
                raise GptContinueException(
                    msg="Empty model output"
                )  # raise exception for empty output
        except GptLengthException:
            deleted_histories += user_gpt_context.ensure_token_not_exceed()
            deleted_histories += user_gpt_context.clear_tokens(
                tokens_to_remove=ChatGPTConfig.extra_token_margin
            )
            continue
        except GptBreakException:
            break
        except GptContinueException:
            generator = llm_client.create_completion(  # type: ignore
                prompt=prompt,
                suffix=llm.suffix,
                max_tokens=min(
                    user_gpt_context.left_tokens,
                    user_gpt_context.gpt_model.value.max_tokens_per_request,
                ),
                temperature=user_gpt_context.user_gpt_profile.temperature,
                top_p=user_gpt_context.user_gpt_profile.top_p,
                logprobs=llm.logprobs,
                echo=bool(llm.echo),
                stop=llm.stop,
                repeat_penalty=user_gpt_context.user_gpt_profile.frequency_penalty,
                top_k=40,
                stream=True,
            )
            continue
        except Exception as e:
            m_queue.put_nowait(e)
            m_done.set()
            return
        else:
            break

    # if content_buffer starts with "user_gpt_context.gpt_profile.gpt_role: " then remove it
    prefix_to_remove: str = f"{user_gpt_context.user_gpt_profile.gpt_role}: "
    if content_buffer.startswith(prefix_to_remove):
        content_buffer = content_buffer[len(prefix_to_remove) :]  # noqa: E203
    m_queue.put(
        {
            "result": {
                "generated_text": content_buffer,
                "n_gen_tokens": len(
                    llm.client.tokenize(b" " + content_buffer.encode("utf-8"))
                ),
                "deleted_histories": deleted_histories,
            }
        }
    )
    m_done.set()  # Notify the main process that we're done


def get_llama(llama_cpp_model: "LlamaCppModel") -> LlamaCpp:
    return LlamaCpp(
        client=None,
        cache=None,
        callbacks=None,
        callback_manager=None,
        model_path=llama_cpp_model.model_path,
        lora_base=llama_cpp_model.lora_base,
        lora_path=llama_cpp_model.lora_path,
        n_ctx=llama_cpp_model.max_total_tokens,
        n_parts=llama_cpp_model.n_parts,
        seed=llama_cpp_model.seed,
        f16_kv=llama_cpp_model.f16_kv,
        logits_all=llama_cpp_model.logits_all,
        vocab_only=llama_cpp_model.vocab_only,
        use_mlock=llama_cpp_model.use_mlock,
        n_threads=llama_cpp_model.n_threads,
        n_batch=llama_cpp_model.n_batch,
        suffix=llama_cpp_model.suffix,
        max_tokens=llama_cpp_model.max_tokens_per_request,
        temperature=llama_cpp_model.temperature,
        top_p=llama_cpp_model.top_p,
        logprobs=llama_cpp_model.logprobs,
        echo=llama_cpp_model.echo,
        stop=llama_cpp_model.stop,
        repeat_penalty=llama_cpp_model.repeat_penalty,
        top_k=llama_cpp_model.top_k,
        last_n_tokens_size=llama_cpp_model.last_n_tokens_size,
        use_mmap=llama_cpp_model.use_mmap,
        streaming=llama_cpp_model.streaming,
    )
