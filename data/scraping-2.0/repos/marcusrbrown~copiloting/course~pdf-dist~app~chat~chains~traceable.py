from langchain.chains.base import Chain
from langfuse.model import CreateTrace

from app.chat.tracing.langfuse import langfuse


class TraceableChain(Chain):
    """A chain that can be traced."""

    def __call__(self, *args, **kwargs):
        if hasattr(self, "metadata"):
            trace = langfuse.trace(
                CreateTrace(
                    id=self.metadata.get("conversation_id"),
                    metadata=self.metadata,
                )
            )

            callbacks = kwargs.get("callbacks", [])
            callbacks.append(trace.get_langchain_handler())
            kwargs["callbacks"] = callbacks

        return super().__call__(*args, **kwargs)
