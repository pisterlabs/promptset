from zrb.helper.typing import List
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.chat_models import ChatOllama, ChatOpenAI
from .schema import ChatModelFactory
from ..task.any_prompt_task import AnyPromptTask
from ..config import DEFAULT_OLLAMA_BASE_URL, DEFAULT_MODEL, OPENAI_API_KEY


def openai_chat_model_factory(
    api_key: str = OPENAI_API_KEY
) -> ChatModelFactory:
    def create_openai_chat_model(task: AnyPromptTask) -> BaseChatModel:
        return ChatOpenAI(
            api_key=task.render_str(api_key),
            callback_manager=task.get_callback_manager(),
            streaming=True
        )
    return create_openai_chat_model


def ollama_chat_model_factory(
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    model: str = DEFAULT_MODEL,
    mirostat: int | str | None = None,
    mirostat_eta: float | str | None = None,
    mirostat_tau: float | str | None = None,
    num_ctx: int | str | None = None,
    num_gpu: int | str | None = None,
    num_thread: int | str | None = None,
    repeat_last_n: int | str | None = None,
    repeat_penalty: float | str | None = None,
    temperature: float | str | None = None,
    stop: List[str] | None = None,
    tfs_z: float | str | None = None,
    top_k: int | str | None = None,
    top_p: int | str | None = None,
    system: str | str | None = None,
    template: str | str | None = None,
    format: str | str | None = None,
    timeout: int | str | None = None,
) -> ChatModelFactory:
    def create_ollama_chat_model(task: AnyPromptTask) -> BaseChatModel:
        return ChatOllama(
            base_url=task.render_str(base_url),
            model=task.render_str(model),
            mirostat=task.render_any(mirostat),
            mirostat_eta=task.render_any(mirostat_eta),
            mirostat_tau=task.render_any(mirostat_tau),
            num_ctx=task.render_any(num_ctx),
            num_gpu=task.render_any(num_gpu),
            num_thread=task.render_any(num_thread),
            repeat_last_n=task.render_any(repeat_last_n),
            repeat_penalty=task.render_any(repeat_penalty),
            temperature=task.render_any(temperature),
            stop=task.render_any(stop),
            tfs_z=task.render_any(tfs_z),
            top_k=task.render_any(top_k),
            top_p=task.render_any(top_p),
            system=task.render_any(system),
            template=task.render_any(template),
            format=task.render_any(format),
            timeout=task.render_any(timeout),
            callback_manager=task.get_callback_manager()
        )
    return create_ollama_chat_model

