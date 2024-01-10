from malevich.square import Context, init

from .ops import LangchainOps


@init(prepare=True)
def initialize_langchain(ctx: Context):
    """Initialize the langchain app.

    Initializes two objects:
        - Embedder (ctx.app_cfg["embedder"]) - used for embeddings
        - Chat (ctx.app_cfg["chat"]) - used for chat

    """
    ctx.common = LangchainOps()
    ctx.common.attach_chat_model(
        backend=ctx.app_cfg.get("chat_backend", "openai"),
        api_key=ctx.app_cfg.get("api_key", None),
        temperature=ctx.app_cfg.get("temperature", 0.5),
    )

    ctx.common.attach_embedder(
        backend=ctx.app_cfg.get("embeddings_backend", "openai"),
        embeddings_type=ctx.app_cfg.get("embeddings_type", "symmetric"),
        model_name=ctx.app_cfg.get("model_name", None),
        api_key=ctx.app_cfg.get("api_key", None),
    )
