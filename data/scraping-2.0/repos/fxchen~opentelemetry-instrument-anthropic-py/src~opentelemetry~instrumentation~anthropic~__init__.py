from typing import Any, Collection, Dict, Optional

import anthropic
import logging

from opentelemetry import trace
from opentelemetry.trace import SpanKind
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.anthropic.package import _instruments
from opentelemetry.instrumentation.anthropic.version import __version__

logger = logging.getLogger(__name__)

SPAN_PREFIX: str = "anthropic"


def no_none(value: Any) -> Any:
    """Replace None with string 'None' for OTEL attributes."""
    return str(value) if value is None else value


def set_span_attributes(
    span, prefix: str, data: Dict[str, Any], suppress_keys: Optional[set] = None
):
    """Set attributes on a span based on a dictionary."""
    for key, value in data.items():
        if suppress_keys and key in suppress_keys:
            continue
        span.set_attribute(f"{prefix}.{key}", no_none(value))


class _InstrumentedAnthropic(anthropic.Anthropic):
    def __init__(
        self,
        tracer_provider: Optional[trace.TracerProvider] = None,
        suppress_input_content: bool = False,
        suppress_response_data: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._tracer_provider = tracer_provider
        self._suppress_input_content = suppress_input_content
        self._suppress_response_data = suppress_response_data
        self.completions.create = self._wrap_completions_create(
            self.completions.create
        )

    def _wrap_completions_create(self, original_func):
        """Wrap 'completions.create' to add telemetry."""

        def wrapper(*args, **kwargs):
            tracer = trace.get_tracer(
                __name__, __version__, self._tracer_provider
            )
            try:
                with tracer.start_as_current_span(
                    f"{SPAN_PREFIX}.completions.create", kind=SpanKind.CLIENT
                ) as span:
                    if span.is_recording():
                        suppress_keys = (
                            {"prompt"} if self._suppress_input_content else None
                        )
                        set_span_attributes(
                            span,
                            f"{SPAN_PREFIX}.input",
                            kwargs,
                            suppress_keys=suppress_keys,
                        )

                    # Handle streaming responses
                    if kwargs.get("stream", False):
                        if span.is_recording():
                            span.set_attribute(
                                f"{SPAN_PREFIX}.response.stream", no_none(True)
                            )
                            span.set_status(Status(StatusCode.OK))
                        return original_func(*args, **kwargs)

                    # Handle standard responses
                    response = original_func(*args, **kwargs)
                    if span.is_recording() and response:
                        suppress_keys = (
                            {"completion"}
                            if self._suppress_response_data
                            else None
                        )
                        set_span_attributes(
                            span,
                            f"{SPAN_PREFIX}.response",
                            vars(response),
                            suppress_keys=suppress_keys,
                        )
                        span.set_status(Status(StatusCode.OK))

                    return response

            except Exception as e:
                logger.error(f"Failed to add span: {e}")
                return original_func(*args, **kwargs)

        return wrapper


class AnthropicInstrumentor(BaseInstrumentor):
    """Instrument Anthropic's client library in Python.

    This class adheres to OpenTelemetry's BaseInstrumentor interface and
    provides automatic instrumentation for Anthropic's Python library.
    """

    def __init__(self):
        self._original_anthropic = anthropic.Anthropic

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        suppress_input_content = kwargs.get("suppress_input_content", False)
        suppress_response_data = kwargs.get("suppress_response_data", False)
        self._replace_anthropic_class(
            tracer_provider, suppress_input_content, suppress_response_data
        )

    def _uninstrument(self, **kwargs):
        self._restore_original_anthropic_class()

    def _replace_anthropic_class(
        self,
        tracer_provider: Optional[trace.TracerProvider] = None,
        suppress_input_content: bool = False,
        suppress_response_data: bool = False,
    ):
        """Replace the original Anthropic class with the instrumented one."""

        self.original_anthropic = (
            anthropic.Anthropic
        )  # Store the original class

        class WrappedAnthropic(_InstrumentedAnthropic):
            def __init__(self, *args, **kwargs):
                super().__init__(
                    tracer_provider=tracer_provider,
                    suppress_input_content=suppress_input_content,
                    suppress_response_data=suppress_response_data,
                    *args,
                    **kwargs,
                )

        WrappedAnthropic.__name__ = "Anthropic"
        anthropic.Anthropic = WrappedAnthropic

    def _restore_original_anthropic_class(self):
        """Restore the original Anthropic class."""
        anthropic.Anthropic = self._original_anthropic

    @staticmethod
    def instrument_instance(
        instance, tracer_provider: Optional[trace.TracerProvider] = None
    ):
        """Instrument a specific instance of the Anthropic class."""
        instance._tracer_provider = tracer_provider
        instance.completions.create = (
            _InstrumentedAnthropic._wrap_completions_create(
                instance.completions.create
            )
        )

    @staticmethod
    def uninstrument_instance(instance):
        """Uninstrument a specific instance of the Anthropic class."""
        instance.completions.create = instance._original_completions_create
