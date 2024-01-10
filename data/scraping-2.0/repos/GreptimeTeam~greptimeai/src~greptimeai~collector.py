import base64
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID

from opentelemetry import metrics, trace
from opentelemetry.context.context import Context
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.metrics import CallbackOptions, Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Status, StatusCode, Tracer, set_span_in_context
from opentelemetry.trace.span import format_span_id, format_trace_id
from opentelemetry.util.types import Attributes, AttributeValue
from typing_extensions import override

from greptimeai import __version__ as greptimeai_version
from greptimeai import logger
from greptimeai.labels import (
    _COMPLETION_COST_LABEL,
    _COMPLETION_TOKENS_LABEL,
    _SOURCE_LABEL,
    _SOURCE_VERSION_LABEL,
    _PROMPT_COST_LABEl,
    _PROMPT_TOKENS_LABEl,
)
from greptimeai.utils._socket import get_local_hostname_and_ip
from greptimeai.utils.json_encoder import CustomEncoder

_JSON_KEYS_IN_OTLP_ATTRIBUTES = "otlp_json_keys"

_GREPTIME_HOST_ENV_NAME = "GREPTIMEAI_HOST"
_GREPTIME_DATABASE_ENV_NAME = "GREPTIMEAI_DATABASE"
_GREPTIME_TOKEN_ENV_NAME = "GREPTIMEAI_TOKEN"


def _prefix_with_scheme_if_not_found(endpoint: Optional[str]) -> Optional[str]:
    if not endpoint:
        return endpoint

    endpoint = endpoint.strip()

    if (
        endpoint == ""
        or endpoint.startswith("https://")
        or endpoint.startswith("http://")
    ):
        return endpoint

    return f"https://{endpoint}"


def _extract_token(token: Optional[str]) -> Tuple[str, str]:
    """
    if token is invalid or empty, then invalid auth header will be included
    """
    if token is None or token.strip() == "":
        return "", ""

    lst = token.split(":", 2)
    if len(lst) != 2:
        return "", ""

    return lst[0], lst[1]


def _check_with_env(
    name: str, value: Optional[str], env_name: str, required: bool = True
) -> str:
    if value and value.strip() != "":
        return value

    value = os.getenv(env_name, "")

    if required and not value:
        raise ValueError(
            f"{name} MUST BE provided either by passing arguments or setup environment variable {env_name}"
        )

    return value


def _is_valid_otel_attributes_value_type(val: Any) -> bool:
    """
    check if value is valid type for OTel attributes
    """
    return isinstance(val, (bool, str, int, float, bytes))


def _sanitate_attributes(attrs: Optional[Dict[str, Any]]) -> Dict[str, AttributeValue]:
    """
    prepare attributes value to any of ['bool', 'str', 'bytes', 'int', 'float']
    or a sequence of these types.

    Other types will be sanitated to json string, and
    append this key in `_JSON_KEYS_IN_OTLP_ATTRIBUTES`
    """
    result: Dict[str, AttributeValue] = {}
    if not attrs:
        return result

    def _json_dumps(val: Any) -> str:
        try:
            return json.dumps(val, cls=CustomEncoder)
        except Exception as e:
            logger.error(f"failed to json.dumps { val } with { e }")
            return str(val)

    def _sanitate_list(lst: list) -> Union[list, str]:
        contains_any_invalid_value_type = False
        for item in lst:
            if not _is_valid_otel_attributes_value_type(item):
                contains_any_invalid_value_type = True
                break
        if contains_any_invalid_value_type:
            return _json_dumps(lst)
        else:
            return lst

    json_keys = []
    for key, val in attrs.items():
        if val is None:
            continue
        elif _is_valid_otel_attributes_value_type(val):
            result[key] = val
        elif isinstance(val, list):
            sanitated_lst = _sanitate_list(val)
            result[key] = sanitated_lst

            if isinstance(sanitated_lst, str):
                json_keys.append(key)
        else:
            result[key] = _json_dumps(val)
            json_keys.append(key)

    if len(json_keys) > 0:
        result[_JSON_KEYS_IN_OTLP_ATTRIBUTES] = json_keys
    return result


class _SpanContext:
    """
    ease context management for OTLP Context and span_id
    """

    def __init__(self, name: str, model: str, span: Span):
        self.name = name
        self.model = model
        self.span = span

    def set_self_as_current(self) -> Context:
        """
        call `get_current_span` from the returning context to retrieve the span
        """
        return set_span_in_context(self.span, Context({}))

    @override
    def __repr__(self) -> str:
        span_context = self.span.get_span_context()
        return f"""
                <span_name={self.name}>
                <model={self.model}>
                <trace_id={span_context.trace_id}>
                <span_id={span_context.span_id}>"""


class _SpanTable:
    """
    NOTE: different span_name may have same span_id.
    """

    def __init__(self):
        self._spans: Dict[str, List[_SpanContext]] = {}

    def put_span_context(self, key: str, context: _SpanContext):
        """
        Pay Attention:

        - different span_name may have same run_id in LangChain.
        - the key may not be the OTLP span_id, maybe it is UUID format from langchain.
          The key is to help find the span_context.
        """
        context_list = self._spans.get(key, [])
        context_list.append(context)
        self._spans[key] = context_list

    def get_span_context(
        self, key: Optional[str], span_name: Optional[str] = None
    ) -> Optional[_SpanContext]:
        """
        Args:

            span_name: if is None or empty, the last context will be returned
        """
        if not key:
            logger.warning(f"get_span_context: key is None for { span_name= }")
            return None

        context_list = self._spans.get(key, [])
        if len(context_list) == 0:
            logger.debug(f"get_span_context: { key } not found for { span_name= }")
            return None

        if not span_name:
            return context_list[-1]

        for context in context_list:
            if span_name == context.name:
                return context

        logger.warning(
            f"get_span_context: { key } not matched for { span_name= }. {context_list}"
        )
        return None

    def pop_span_context(
        self, key: Optional[str], span_name: Optional[str] = None
    ) -> Optional[_SpanContext]:
        """
        if there is only one span matched this span_id, then span_id key will be removed

        Args:

            span_name: if is None or empty, the last context will be returned
        """
        if not key:
            logger.warning(f"pop_span_context: key is None for { span_name= }")
            return None

        context_list = self._spans.get(key, [])
        if len(context_list) == 0:
            logger.debug(f"pop_span_context: { key } not found for { span_name= }")
            return None

        target_context = None
        rest_list = []

        if not span_name:
            target_context = context_list.pop()
            rest_list = context_list
        else:
            for context in context_list:
                if span_name == context.name:
                    target_context = context
                else:
                    rest_list.append(context)

        if len(rest_list) == 0:
            self._spans.pop(key, None)
        else:
            self._spans[key] = rest_list

        if not target_context:
            logger.warning(
                f"pop_span_context: { key } not matched for { span_name= }. {context_list}"
            )

        return target_context

    @override
    def __repr__(self) -> str:
        return f"{self._spans}"


class _DurationTable:
    def __init__(self):
        self._tables: Dict[str, float] = {}

    def _key(self, span_id: Union[UUID, str], name: Optional[str]) -> str:
        if name:
            return f"{name}.{span_id}"
        else:
            return str(span_id)

    def set(self, span_id: Union[UUID, str], name: Optional[str] = None):
        """
        set start time of span_id.

        Args:

            name: same span may have different name, this is to differentiate them.
        """
        key = self._key(span_id, name)
        self._tables[key] = time.time()

    def latency_in_ms(
        self, span_id: Union[UUID, str], name: Optional[str] = None
    ) -> Optional[float]:
        """
        return latency in millisecond if key exist, None if not exist.
        """
        key = self._key(span_id, name)
        start = self._tables.pop(key, None)
        if start:
            now = time.time()
            return 1000 * (now - start)
        return None

    @override
    def __repr__(self) -> str:
        return f"{self._tables}"


class _Observation:
    """
    # FIXME(yuanbohan): concurrent conflict. refer to Mutex or other solutions
    """

    def __init__(self, name: str):
        self._name = name
        self._value: Dict[Tuple, float] = {}

    def _reset(self):
        self._value = {}

    def _attrs_to_tuple(self, attrs: Optional[Attributes] = None) -> Tuple:
        """
        sort keys first
        """
        if attrs is None or len(attrs) == 0:
            return ()

        lst = [(key, attrs[key]) for key in sorted(attrs.keys())]
        return tuple(lst)

    def put(self, val: float, attributes: Optional[Attributes] = None):
        """
        if attrs is None or empty, nothing will happen
        """
        if attributes is None or len(attributes) == 0:
            logger.info(f"None key for { attributes }")
            return

        tuple_key = self._attrs_to_tuple(attributes)
        self._value.setdefault(tuple_key, 0)
        self._value[tuple_key] += val

    def observation_callback(self):
        """
        _cost will be reset each time being called
        """

        def callback(_: CallbackOptions):
            lst = [
                Observation(val, dict(tuple_key))
                for tuple_key, val in self._value.items()
            ]
            self._reset()
            return lst

        return callback


class OTel:
    def __init__(self) -> None:
        self.is_setup = False

    def setup(
        self,
        host: str = "",
        database: str = "",
        token: str = "",
    ):
        if self.is_setup:
            logger.warning("otel is already setup, skip")
            return

        self.host = _prefix_with_scheme_if_not_found(
            _check_with_env("host", host, _GREPTIME_HOST_ENV_NAME, True)
        )
        self.database = _check_with_env(
            "database", database, _GREPTIME_DATABASE_ENV_NAME, True
        )
        self.token = _check_with_env("token", token, _GREPTIME_TOKEN_ENV_NAME, False)

        self._duration_tables = _DurationTable()
        self._prompt_cost = _Observation("prompt_cost")
        self._completion_cost = _Observation("completion_cost")
        self._span_tables = _SpanTable()

        self._setup_otel_exporter()
        self._setup_otel_metrics()

        self.is_setup = True

    def _setup_otel_exporter(self):
        hostname, ip = get_local_hostname_and_ip()
        resource = Resource.create(
            {
                SERVICE_NAME: "greptimeai",
                SERVICE_VERSION: greptimeai_version,
                "host.name": hostname,
                "host.ip": ip,
            }
        )
        metrics_endpoint = f"{self.host}/v1/otlp/v1/metrics"
        trace_endpoint = f"{self.host}/v1/otlp/v1/traces"

        username, password = _extract_token(self.token)
        auth = f"{username}:{password}"
        b64_auth = base64.b64encode(auth.encode()).decode("ascii")
        greptime_headers: Dict[str, str] = {
            "Authorization": f"Basic {b64_auth}",
            "x-greptime-db-name": self.database,
        }

        metrics_exporter = OTLPMetricExporter(
            endpoint=metrics_endpoint,
            headers=greptime_headers,
            timeout=5,
        )
        self._metric_reader = PeriodicExportingMetricReader(
            metrics_exporter, export_interval_millis=15000
        )
        meter_provider = MeterProvider(
            resource=resource, metric_readers=[self._metric_reader]
        )
        metrics.set_meter_provider(meter_provider)

        trace_provider = TracerProvider(resource=resource)
        self._span_processor = BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=trace_endpoint,
                headers=greptime_headers,
                timeout=5,
            )
        )
        trace_provider.add_span_processor(self._span_processor)
        trace.set_tracer_provider(trace_provider)

        self._tracer = trace.get_tracer(
            instrumenting_module_name="greptimeai",
            instrumenting_library_version=greptimeai_version,
        )

    @property
    def tracer(self) -> Tracer:
        return self._tracer

    def _setup_otel_metrics(self):
        """
        setup opentelemetry, and raise Error if something wrong
        """

        meter = metrics.get_meter(name="greptimeai", version=greptimeai_version)

        self._prompt_tokens_count = meter.create_counter(
            "llm_prompt_tokens",
            description="counts the amount of llm prompt token",
        )

        self._completion_tokens_count = meter.create_counter(
            "llm_completion_tokens",
            description="counts the amount of llm completion token",
        )

        self._requests_duration_histogram = meter.create_histogram(
            name="llm_request_duration_ms",
            description="duration of requests of llm in milliseconds",
            unit="ms",
        )

        meter.create_observable_gauge(
            callbacks=[self._prompt_cost.observation_callback()],
            name="llm_prompt_tokens_cost",
            description="prompt token cost in US Dollar",
        )

        meter.create_observable_gauge(
            callbacks=[self._completion_cost.observation_callback()],
            name="llm_completion_tokens_cost",
            description="completion token cost in US Dollar",
        )

    def _force_flush(self):
        """
        DO NOT call this method to flush metrics and traces, this is only for test cases.
        """
        self._metric_reader.force_flush()
        self._span_processor.force_flush()


otel: OTel = OTel()  # only one instance of OTel is allowed


class Collector:
    """
    Collector class is responsible for collecting metrics and traces.

    Args:
        host (str, optional): The host URL. Defaults to "".
        database (str, optional): The name of the database. Defaults to "".
        token (str, optional): The authentication token. Defaults to "".

        source (str): The source of the collector. openai, langchain so far.
        version (str): The version of the source.
    """

    def __init__(
        self,
        host: str = "",
        database: str = "",
        token: str = "",
        source: str = "unknown",
        source_version: str = "unknown",
    ):
        self.source = source
        self.source_version = source_version
        otel.setup(host=host, database=database, token=token)

    def start_span(
        self,
        span_id: Union[UUID, str, None],  # uuid is from langchain
        parent_id: Union[UUID, str, None],  # uuid is from langchain
        span_name: str,
        event_name: str,
        span_attrs: Dict[str, Any] = {},  # model SHOULD exist in span attrs
        event_attrs: Dict[str, Any] = {},
    ) -> Tuple[str, str]:
        """
        NOTE: end_span MUST BE called with the the same span_id and span_name to revoke the key in trace table.

        if span_id is None when calling start_span, then use the returned span_id for end_span.

        Args:

            span_id: span id. If None, this id will be automatically generated.
            parent_id: parent span id. if None, this is a root span.

        Returns:

            Tuple of trace_id and span_id.
            And the returned span_id is to help retrieve the span_context from otel._span_tables

        """
        span_attrs = span_attrs or {}
        event_attrs = event_attrs or {}
        span_id = str(span_id) if span_id else None
        parent_id = str(parent_id) if parent_id else None
        logger.debug(f"start span for {span_name=} with {span_id=} or {parent_id=}")
        span_attributes = _sanitate_attributes(
            {
                _SOURCE_LABEL: self.source,
                _SOURCE_VERSION_LABEL: self.source_version,
                **span_attrs,
            }
        )
        event_attributes = _sanitate_attributes(event_attrs)

        def _do_start_span(ctx: Optional[Context] = None) -> Tuple[str, str]:
            span = otel.tracer.start_span(
                span_name, context=ctx, attributes=span_attributes
            )
            span.add_event(event_name, attributes=event_attributes)

            trace_context = _SpanContext(
                name=span_name, model=span_attrs.get("model", ""), span=span
            )

            span_context = span.get_span_context()
            otel_trace_id = format_trace_id(span_context.trace_id)
            otel_span_id = format_span_id(span_context.span_id)

            final_span_id = str(span_id) if span_id else otel_span_id
            otel._span_tables.put_span_context(final_span_id, trace_context)
            return otel_trace_id, final_span_id

        # start span and let otel generate the trace id and span id
        if not parent_id and not span_id:
            return _do_start_span(None)

        if parent_id:  # if parent_id, then start a child span
            span_context = otel._span_tables.get_span_context(parent_id)
            if span_context:
                return _do_start_span(span_context.set_self_as_current())
            else:
                logger.error(
                    f"Parent span of { parent_id } not found, will act as Root span."
                )
                return _do_start_span()
        else:
            # span_context may exist for the same run_id in LangChain. For Example:
            # different Chain triggered in the same trace may have the same run_id.
            span_context = otel._span_tables.get_span_context(span_id)
            if span_context:
                return _do_start_span(span_context.set_self_as_current())
            else:
                return _do_start_span()

    def add_span_event(
        self, span_id: Union[UUID, str], event_name: str, event_attrs: Dict[str, Any]
    ):
        """
        this is mainly focused on LangChain.
        """
        logger.debug(f"add event for {event_name} with {span_id=}")
        attrs = _sanitate_attributes(event_attrs)
        context = otel._span_tables.get_span_context(str(span_id))
        if context:
            context.span.add_event(event_name, attributes=attrs)
        else:
            logger.error(f"{span_id} span not found for {event_name}")

    def end_span(
        self,
        span_id: Union[UUID, str, None],
        span_name: str,
        event_name: str,
        span_attrs: Dict[str, Any] = {},
        event_attrs: Dict[str, Any] = {},
        ex: Optional[BaseException] = None,
    ):
        span_attrs = span_attrs or {}
        event_attrs = event_attrs or {}
        span_id = str(span_id) if span_id else None
        logger.debug(f"end span for {span_name} with {span_id=}")

        span_attributes = _sanitate_attributes(
            {_SOURCE_LABEL: self.source, **span_attrs}
        )
        event_attributes = _sanitate_attributes(event_attrs)

        context = otel._span_tables.pop_span_context(span_id, span_name)
        logger.debug(f"end span for {span_id=} {span_name=} with {context=}")
        if context and context.span:
            span = context.span
            if ex:
                span.record_exception(ex)  # type: ignore
            if span_attributes:
                span.set_attributes(attributes=span_attributes)
            code = StatusCode.ERROR if ex else StatusCode.OK
            span.set_status(Status(code))
            span.add_event(event_name, attributes=event_attributes)
            span.end()
        else:
            logger.error(
                f"unexpected behavior of end_span. context of { span_id } and { span_name } not found."
            )

    def collect_metrics(self, span_attrs: Dict[str, Any], attrs: Optional[Attributes]):
        """
        Collects metrics for the given span attributes and optional attributes.

        Args:
            span_attrs (Dict[str, Any]): The span attributes containing the metrics data.
            attrs (Optional[Attributes]): Optional additional attributes.

        Notes:
            The `attrs` for metrics should be small enough to keep the series small.
            Detailed information should be stored in traces instead of metrics.
        """
        prompt_tokens = span_attrs.get(_PROMPT_TOKENS_LABEl, 0)
        prompt_cost = span_attrs.get(_PROMPT_COST_LABEl, 0)
        completion_tokens = span_attrs.get(_COMPLETION_TOKENS_LABEL, 0)
        completion_cost = span_attrs.get(_COMPLETION_COST_LABEL, 0)

        self._collect_metrics(
            prompt_tokens=prompt_tokens,
            prompt_cost=prompt_cost,
            completion_tokens=completion_tokens,
            completion_cost=completion_cost,
            attrs=attrs,
        )

    def _collect_metrics(
        self,
        prompt_tokens: int,
        prompt_cost: float,
        completion_tokens: int,
        completion_cost: float,
        attrs: Optional[Attributes] = None,
    ):
        attrs = attrs or {}
        attributes = {_SOURCE_LABEL: self.source, **attrs}

        if prompt_tokens:
            otel._prompt_tokens_count.add(prompt_tokens, attributes=attributes)

        if prompt_cost:
            otel._prompt_cost.put(prompt_cost, attributes=attributes)

        if completion_tokens:
            otel._completion_tokens_count.add(completion_tokens, attributes=attributes)

        if completion_cost:
            otel._completion_cost.put(completion_cost, attributes=attributes)

    def start_latency(self, span_id: Union[UUID, str], span_name: Optional[str]):
        """
        if latency can not be calculated, call start_latency with span_id and span_name,
        then call end_latency with the same span_id and span_name.

        NOTE: end_latency MUST BE called to revoke the key in duration table.
        """
        otel._duration_tables.set(span_id, span_name)

    def end_latency(
        self,
        span_id: Union[UUID, str],
        span_name: Optional[str],
        attributes: Dict[str, Any],
    ):
        latency = otel._duration_tables.latency_in_ms(span_id, span_name)
        self.record_latency(latency, attributes)

    def record_latency(
        self, latency: Union[None, int, float], attrs: Optional[Attributes] = None
    ):
        """
        directly record latency in millisecond

        Args:

            latency: millisecond unit
        """
        if latency:
            attrs = attrs or {}
            attributes = {_SOURCE_LABEL: self.source, **attrs}

            otel._requests_duration_histogram.record(latency, attributes)
        else:
            logger.warning(
                f"latency won't be recorded for None value. attribute is: { attrs }"
            )

    def discard_latency(self, span_id: Union[UUID, str], span_name: Optional[str]):
        otel._duration_tables.latency_in_ms(span_id, span_name)

    def get_model_in_context(self, span_id: Union[UUID, str]) -> Optional[str]:
        context = otel._span_tables.get_span_context(str(span_id))

        if context:
            return context.model

        return None
