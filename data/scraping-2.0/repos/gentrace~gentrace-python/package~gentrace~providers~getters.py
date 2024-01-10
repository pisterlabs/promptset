import os

import openai
from urllib3.util import parse_url

from gentrace.configuration import Configuration as GentraceConfiguration
from gentrace.providers.init import GENTRACE_CONFIG_STATE
from gentrace.providers.utils import is_openai_v1

openai.api_key = os.getenv("OPENAI_KEY")


def test_validity():
    from gentrace import api_key, host

    if not api_key and not GENTRACE_CONFIG_STATE["GENTRACE_API_KEY"]:
        raise ValueError("Gentrace API key not set. Call the init() function first.")

    # Totally fine (and expected) to not have a host set
    if not host:
        return

    path = parse_url(host).path

    if host and path != "/api" and path != "/api/":
        raise ValueError("Gentrace host is invalid")


def configure_openai():
    from gentrace import api_key, host

    test_validity()

    if api_key:
        resolved_host = host if host else "https://gentrace.ai/api"
        gentrace_config = GentraceConfiguration(host=resolved_host)
        gentrace_config.access_token = api_key
    else:
        gentrace_config = GENTRACE_CONFIG_STATE["global_gentrace_config"]

    if not is_openai_v1():
        from .llms.openai_v0 import annotate_openai_module
        annotate_openai_module(gentrace_config=gentrace_config)


def configure_pinecone():
    from .vectorstores.pinecone import annotate_pinecone_module

    test_validity()

    annotate_pinecone_module()


__all__ = ["configure_openai", "configure_pinecone"]
