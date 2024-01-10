from typing import Any, Optional

from .tracer import OpenInferenceTracer
from opentelemetry.trace import get_tracer
from opentelemetry.instrumentation.langchain.version import __version__
from opentelemetry.metrics import (
    CallbackOptions,
    Observation,
    get_meter_provider,
    set_meter_provider,
)


class LangChainHandlerInstrumentor:
    """
    Instruments the OpenInferenceTracer for LangChain automatically by patching the
    BaseCallbackManager in LangChain.
    """

    def __init__(self, handeler: Optional[OpenInferenceTracer] = None) -> None:
        self._handeler = handeler if handeler is not None else OpenInferenceTracer()

    def instrument(self, *args, **kwargs) -> None:
        try:
            from langchain.callbacks.base import BaseCallbackManager
        except ImportError:
            # Raise a cleaner error if LangChain is not installed
            raise ImportError(
                "LangChain is not installed. Please install LangChain first to use the instrumentor"
            )
        tracer_provider = kwargs.get("tracer_provider", None)
        tracer = get_tracer(__name__, __version__, tracer_provider)
        self._handeler.tracer = tracer
        metric_provider = kwargs.get("metric_provider", None)
        # tracer = get_tracer(__name__, __version__, tracer_provider)
        meter = metric_provider.get_meter(__name__, __version__)
        self._handeler.meter = meter
        
        source_init = BaseCallbackManager.__init__

        # Keep track of the source init so we can tell if the patching occurred
        self._source_callback_manager_init = source_init

        handeler = self._handeler

        # Patch the init method of the BaseCallbackManager to add the tracer
        # to all callback managers
        def patched_init(self: BaseCallbackManager, *args: Any, **kwargs: Any) -> None:
            source_init(self, *args, **kwargs)
            self.add_handler(handeler, True)

        BaseCallbackManager.__init__ = patched_init
