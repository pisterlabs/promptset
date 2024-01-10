import os
from types import SimpleNamespace
from typing import Dict, Optional, Tuple
from urllib.parse import urljoin

import openai
from openai import api_key

from .utils import get_aws_auth, get_base_url, get_user

_base_url = get_base_url(api_key.split("-")[1])
_custom_auth = get_aws_auth()
_custom_headers = {
    "openai-proxy-user": get_user(),
    "openai-proxy-project": os.environ.get("OPENAI_DEFAULT_PROJECT", "N/A"),
    "openai-proxy-staging": os.environ.get("OPENAI_DEFAULT_STAGING", "dev"),
    "openai-proxy-caching": os.environ.get("OPENAI_DEFAULT_CACHING", "1"),
}


def set_project(project: str):
    _custom_headers["openai-proxy-project"] = project


def set_staging(staging: str):
    _custom_headers["openai-proxy-staging"] = staging


def set_caching(caching: bool):
    _custom_headers["openai-proxy-caching"] = str(int(caching))


_prepare_request_raw = openai.api_requestor.APIRequestor._prepare_request_raw


def _prepare_request_raw_proxy(
    self,
    url,
    supplied_headers,
    method,
    params,
    files,
    request_id: Optional[str],
) -> Tuple[str, Dict[str, str], Optional[bytes]]:
    _, headers, data = _prepare_request_raw(
        self, url, supplied_headers, method, params, files, request_id
    )

    request = _custom_auth(
        SimpleNamespace(
            **{
                "method": method,
                "url": urljoin(_base_url, url),
                "headers": {**headers, **_custom_headers},
                "content": data,
            }
        )
    )
    return request.url, request.headers, request.content


# Monkey patch
openai.api_requestor.APIRequestor._prepare_request_raw = _prepare_request_raw_proxy
