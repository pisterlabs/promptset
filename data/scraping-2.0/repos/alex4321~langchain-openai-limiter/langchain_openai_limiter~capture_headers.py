"""
Module which set hooks to catch limit-related headers from the OpenAI response
"""
from datetime import datetime, timedelta
import json
from typing import Callable, Tuple, Union
import aiohttp
import openai
import openai.api_requestor
import requests
from .reset_time_parser import reset_time_to_ms
from .limit_info import OrganizationLimitInfo, ApiKey, ModelName, set_limit_info, aset_limit_info


def _extract_openai_api_key(authorization: str) -> ApiKey:
    """
    Extract API key from authorization string
    """
    assert authorization.startswith("Bearer")
    _, openai_api_key = authorization.split(" ")
    return openai_api_key


def _extract_limit_info(headers: dict) -> Tuple[Union[None, ModelName],\
                                                OrganizationLimitInfo]:
    """
    Parse limit information inside headers dictionary
    :return: Pair of model name + limit info
    """
    current_time = datetime.now()
    model_name: ModelName = headers.get("openai-model")
    rpm_total = int(headers["x-ratelimit-limit-requests"])
    tpm_total = int(headers["x-ratelimit-limit-tokens"])
    rpm_remain = int(headers["x-ratelimit-remaining-requests"])
    tpm_remain = int(headers["x-ratelimit-remaining-tokens"])
    rpm_reset_ms = reset_time_to_ms(headers["x-ratelimit-reset-requests"])
    tpm_reset_ms = reset_time_to_ms(headers["x-ratelimit-reset-tokens"])
    rpm_reset_time = current_time + timedelta(milliseconds=rpm_reset_ms)
    tpm_reset_time = current_time + timedelta(milliseconds=tpm_reset_ms)
    return model_name, OrganizationLimitInfo(
        tpm_total=tpm_total,
        tpm_remain=tpm_remain,
        rpm_total=rpm_total,
        rpm_remain=rpm_remain,
        rpm_reset_time=rpm_reset_time,
        tpm_reset_time=tpm_reset_time,
    )


# region Sync stuff
_ATTACHED_SYNC_SESSION_HOOKS = False


# pylint: disable=unused-argument
def _response_hook(response: requests.Response, *args, **kwargs) -> None:
    """
    Hook for `requests` session
    """
    api_key = _extract_openai_api_key(response.request.headers["authorization"])
    model_name, limit_info = _extract_limit_info(response.headers)
    if model_name is None:
        model_name = json.loads(response.request.body).get("model")
    assert model_name is not None
    set_limit_info(model_name, api_key, limit_info)
# pylint: enable=unused-argument


def _attach_to_session(session: requests.Session) -> None:
    """
    Attach `_response_hook` hook to `requests` session
    """
    original_hook = session.hooks.get("response")
    if not isinstance(original_hook, list):
        original_hook = [original_hook]
    session.hooks["response"] = original_hook + [_response_hook]


def _attach_to_session_getter(getter: Callable[[], requests.Session]) -> \
    Callable[[], requests.Session]:
    """
    Attach `_response_hook` hook to `requests` session generator
    """
    def new_getter() -> requests.Session:
        session = getter()
        _attach_to_session(session)
        return session

    return new_getter


def _attach_sync_session_hooks():
    """
    Attach synchronyous code hooks
    """
    # pylint: disable=global-statement
    global _ATTACHED_SYNC_SESSION_HOOKS
    # pylint: enable=global-statement
    if not _ATTACHED_SYNC_SESSION_HOOKS:
        _ATTACHED_SYNC_SESSION_HOOKS = True
        if openai.requestssession is None:
            openai.requestssession = requests.Session()
        if isinstance(openai.requestssession, requests.Session):
            _attach_to_session(openai.requestssession)
        elif callable(openai.requestssession):
            openai.requestssession = _attach_to_session_getter(openai.requestssession)
# endregion


# region Async stuff
_ATTACHED_ASYNC_SESSION_HOOKS = False


def _wrap_arequest_raw(old_arequest_raw):
    """
    Wrap old `openai.api_requestor.APIRequestor.arequest_raw` in
    a new decorator able to call our hook
    :param old_arequest_raw: Original `openai.api_requestor.APIRequestor.arequest_raw`
    :return: decorated `openai.api_requestor.APIRequestor.arequest_raw`
    """
    async def arequest_raw(self, *args, **kwargs) -> aiohttp.ClientResponse:
        response: aiohttp.ClientResponse = await old_arequest_raw(self, *args, **kwargs)
        api_key = _extract_openai_api_key(response.request_info.headers["authorization"])
        model_name, limit_info = _extract_limit_info(response.headers)
        if model_name is None:
            model_name = response.request_info.headers.get("x-model")
        assert model_name is not None
        await aset_limit_info(model_name, api_key, limit_info)
        return response

    return arequest_raw


def _attach_async_session_hooks():
    """
    Attach limit tracking hooks to async calls
    """
    # pylint: disable=global-statement
    global _ATTACHED_ASYNC_SESSION_HOOKS
    # pylint: enable=global-statement
    if not _ATTACHED_ASYNC_SESSION_HOOKS:
        _ATTACHED_ASYNC_SESSION_HOOKS = True
        old_arequest_raw = openai.api_requestor.APIRequestor.arequest_raw
        new_arequest_raw = _wrap_arequest_raw(old_arequest_raw)
        openai.api_requestor.APIRequestor.arequest_raw = new_arequest_raw
# endregion


def attach_session_hooks():
    """
    Attach both synchronyous and asynchronyous hooks
    """
    _attach_sync_session_hooks()
    _attach_async_session_hooks()
