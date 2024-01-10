import os
from typing import Generator
from unittest.mock import call, Mock, create_autospec, AsyncMock

import openai
import pytest
from dotenv import load_dotenv
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from baserun import Baserun, baserun
from baserun.grpc import (
    get_or_create_async_submission_service,
    get_or_create_submission_service,
)
from baserun.v1.baserun_pb2 import Run, Span, StartRunRequest

load_dotenv()


@pytest.fixture(autouse=True)
def clear_context():
    baserun.baserun_contexts = {}


@pytest.fixture(autouse=True)
def set_openai_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")

    openai.api_key = api_key


@pytest.fixture
def mock_services() -> Generator[dict[str, Mock], None, None]:
    # First, create the services
    get_or_create_async_submission_service()
    get_or_create_submission_service()

    services = {
        "submission_service": create_autospec,
        "async_submission_service": AsyncMock,
    }
    rpcs = [
        "EndRun",
        "EndSession",
        "EndTestSuite",
        "GetTemplates",
        "StartRun",
        "StartSession",
        "StartTestSuite",
        "SubmitEval",
        "SubmitLog",
        "SubmitModelConfig",
        "SubmitAnnotations",
        "SubmitSpan",
        "SubmitTemplateVersion",
        "SubmitUser",
    ]

    # Create a dictionary to hold the mocks
    mock_dict = {}

    # Create mocks for each service
    for service, mocking_fn in services.items():
        original_service = getattr(Baserun, service)
        if mocking_fn == AsyncMock:
            mock_service = mocking_fn(spec=original_service)
        else:
            mock_service = mocking_fn(original_service, instance=True)

        setattr(Baserun, service, mock_service)
        mock_dict[service] = mock_service

        # Mock each RPC method in the service
        for rpc in rpcs:
            rpc_attr = getattr(original_service, rpc, None)

            if mocking_fn == AsyncMock:
                mock_method = mocking_fn(spec=rpc_attr)
                mock_method.future = mocking_fn(spec=rpc_attr)
            else:
                mock_method = mocking_fn(rpc_attr, instance=True)

            setattr(mock_service, rpc, mock_method)

    # Yield the dictionary of mock services
    yield mock_dict

    # Remove the mocked instances so they'll be recreated fresh in the next test
    Baserun.submission_service = None
    Baserun.async_submission_service = None


def pytest_sessionstart(session):
    """Starting up Baserun in tests requires that these things happen in a specific order:
    - `init`, specifically setting up gRPC
    - Mock services, to replace the services that were just set up
    - Instrument
    - Close channel, simply to ensure that no unmocked calls get through
    """
    Baserun.init(instrument=False)
    # mock_services()
    # Replace the batch processor so that things happen synchronously and not in a separate thread
    Baserun.instrument(processor_class=SimpleSpanProcessor)


def get_mock_objects(mock_services) -> tuple[Run, Span, Run, Run]:
    mock_start_run = mock_services["submission_service"].StartRun.future
    mock_end_run = mock_services["submission_service"].EndRun.future
    mock_submit_span = mock_services["submission_service"].SubmitSpan

    mock_start_run.assert_called_once()
    run_call: call = mock_start_run.call_args_list[0]
    start_run_request: StartRunRequest = run_call.args[0]
    started_run = start_run_request.run

    if mock_submit_span.call_args_list:
        mock_submit_span.assert_called_once()
        submit_span_call: call = mock_submit_span.call_args_list[0]
        submit_span_request = submit_span_call.args[0]
        span = submit_span_request.span
        submitted_run = submit_span_request.run
    else:
        span = None
        submitted_run = None

    mock_end_run.assert_called_once()
    end_run_call: call = mock_end_run.call_args_list[0]
    end_run_request: StartRunRequest = end_run_call.args[0]
    ended_run = end_run_request.run

    return started_run, span, submitted_run, ended_run
