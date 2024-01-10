import asyncio
import json
import platform
import sys
import time
import warnings
from json import JSONDecodeError
from typing import AsyncGenerator, Callable, Dict, Iterator, Optional, Tuple, Union, overload, Generator, Coroutine, Any
from urllib.parse import urlencode, urlsplit, urlunsplit

import httpx
import requests

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

import openai
from openai import error, util, version
from openai.openai_response import OpenAIResponse
from openai.util import ApiType

TIMEOUT_SECS = 600
MAX_SESSION_LIFETIME_SECS = 180
MAX_CONNECTION_RETRIES = 2


def _build_api_url(url, query):
    scheme, netloc, path, base_query, fragment = urlsplit(url)

    if base_query:
        query = "%s&%s" % (base_query, query)

    return urlunsplit((scheme, netloc, path, query, fragment))


def _requests_proxies_arg(proxy) -> Optional[Dict[str, str]]:
    """Returns a value suitable for the 'proxies' argument to 'requests.request."""
    if proxy is None:
        return None
    elif isinstance(proxy, str):
        return {"http": proxy, "https": proxy}
    elif isinstance(proxy, dict):
        return proxy.copy()
    else:
        raise ValueError(
            "'openai.proxy' must be specified as either a string URL or a dict with string URL under the https and/or http keys."
        )


def init_session(sync: bool = True) -> Union[httpx.Client, httpx.AsyncClient]:
    if not openai.verify_ssl_certs:
        warnings.warn("verify_ssl_certs is ignored; openai always verifies.")

    proxies = _requests_proxies_arg(openai.proxy)

    client_config = {
        "verify": openai.verify_ssl_certs,
    }

    if proxies:
        client_config["proxies"] = proxies
    if sync:
        return httpx.Client(**client_config)
    return httpx.AsyncClient(**client_config)


class APIRequestor:
    def __init__(
        self,
        key=None,
        api_base=None,
        api_type=None,
        api_version=None,
        organization=None,
    ):
        self.api_base = api_base or openai.api_base
        self.api_key = key or util.default_api_key()
        self.api_type = ApiType.from_str(api_type) if api_type else ApiType.from_str(openai.api_type)
        self.api_version = api_version or openai.api_version
        self.organization = organization or openai.organization
        self.sync_session = openai.sync_session
        self.async_session = openai.async_session

    @classmethod
    def format_app_info(cls, info):
        str = info["name"]
        if info["version"]:
            str += "/%s" % (info["version"],)
        if info["url"]:
            str += " (%s)" % (info["url"],)
        return str

    def _check_polling_response(self, response: OpenAIResponse, predicate: Callable[[OpenAIResponse], bool]):
        if not predicate(response):
            return
        error_data = response.data["error"]
        message = error_data.get("message", "Operation failed")
        code = error_data.get("code")
        raise error.OpenAIError(message=message, code=code)

    def _poll(
        self, method, url, until, failed, params=None, headers=None, interval=None, delay=None
    ) -> Tuple[Iterator[OpenAIResponse], bool, str]:
        if delay:
            time.sleep(delay)

        response, b, api_key = self.request(method, url, params, headers)
        self._check_polling_response(response, failed)
        start_time = time.time()
        while not until(response):
            if time.time() - start_time > TIMEOUT_SECS:
                raise error.Timeout("Operation polling timed out.")

            time.sleep(interval or response.retry_after or 10)
            response, b, api_key = self.request(method, url, params, headers)
            self._check_polling_response(response, failed)

        response.data = response.data["result"]
        return response, b, api_key

    async def _apoll(
        self, method, url, until, failed, params=None, headers=None, interval=None, delay=None
    ) -> Tuple[Iterator[OpenAIResponse], bool, str]:
        if delay:
            await asyncio.sleep(delay)

        response, b, api_key = await self.arequest(method, url, params, headers)
        self._check_polling_response(response, failed)
        start_time = time.time()
        while not until(response):
            if time.time() - start_time > TIMEOUT_SECS:
                raise error.Timeout("Operation polling timed out.")

            await asyncio.sleep(interval or response.retry_after or 10)
            response, b, api_key = await self.arequest(method, url, params, headers)
            self._check_polling_response(response, failed)

        response.data = response.data["result"]
        return response, b, api_key

    @overload
    def request(
        self,
        method,
        url,
        params,
        headers,
        files,
        stream: Literal[True],
        request_id: Optional[str] = ...,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = ...,
    ) -> Tuple[Iterator[OpenAIResponse], bool, str]:
        pass

    @overload
    def request(
        self,
        method,
        url,
        params=...,
        headers=...,
        files=...,
        *,
        stream: Literal[True],
        request_id: Optional[str] = ...,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = ...,
    ) -> Tuple[Iterator[OpenAIResponse], bool, str]:
        pass

    @overload
    def request(
        self,
        method,
        url,
        params=...,
        headers=...,
        files=...,
        stream: Literal[False] = ...,
        request_id: Optional[str] = ...,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = ...,
    ) -> Tuple[OpenAIResponse, bool, str]:
        pass

    @overload
    def request(
        self,
        method,
        url,
        params=...,
        headers=...,
        files=...,
        stream: bool = ...,
        request_id: Optional[str] = ...,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = ...,
    ) -> Tuple[Union[OpenAIResponse, Iterator[OpenAIResponse]], bool, str]:
        pass

    def request(
        self,
        method,
        url,
        params=None,
        headers=None,
        files=None,
        stream: bool = False,
        request_id: Optional[str] = None,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = None,
    ) -> Tuple[Union[OpenAIResponse, Iterator[OpenAIResponse]], bool, str]:
        result = self.request_raw(
            method.lower(),
            url,
            params=params,
            supplied_headers=headers,
            files=files,
            stream=stream,
            request_id=request_id,
            request_timeout=request_timeout,
        )
        resp, got_stream = self._interpret_response(result, stream)
        return resp, got_stream, self.api_key

    @overload
    async def arequest(
        self,
        method,
        url,
        params,
        headers,
        files,
        stream: Literal[True],
        request_id: Optional[str] = ...,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = ...,
    ) -> Tuple[AsyncGenerator[OpenAIResponse, None], bool, str]:
        pass

    @overload
    async def arequest(
        self,
        method,
        url,
        params=...,
        headers=...,
        files=...,
        *,
        stream: Literal[True],
        request_id: Optional[str] = ...,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = ...,
    ) -> Tuple[AsyncGenerator[OpenAIResponse, None], bool, str]:
        pass

    @overload
    async def arequest(
        self,
        method,
        url,
        params=...,
        headers=...,
        files=...,
        stream: Literal[False] = ...,
        request_id: Optional[str] = ...,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = ...,
    ) -> Tuple[OpenAIResponse, bool, str]:
        pass

    @overload
    async def arequest(
        self,
        method,
        url,
        params=...,
        headers=...,
        files=...,
        stream: bool = ...,
        request_id: Optional[str] = ...,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = ...,
    ) -> Tuple[Union[OpenAIResponse, AsyncGenerator[OpenAIResponse, None]], bool, str]:
        pass

    async def arequest(
        self,
        method,
        url,
        params=None,
        headers=None,
        files=None,
        stream: bool = False,
        request_id: Optional[str] = None,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = None,
    ) -> Tuple[Union[OpenAIResponse, AsyncGenerator[OpenAIResponse, None]], bool, str]:
        resp = None
        try:
            result = await self.arequest_raw(
                method.lower(),
                url,
                self.async_session,
                stream=stream,
                params=params,
                supplied_headers=headers,
                files=files,
                request_id=request_id,
                request_timeout=request_timeout,
            )
            if not stream:
                resp, got_stream = self._interpret_response(result, stream)
            else:
                got_stream = True

        except Exception:
            # Close the request before exiting session context.
            raise
        if got_stream:

            async def wrap_resp():
                try:
                    yield_response = []
                    async for text in result.aiter_bytes():
                        text = text.decode("utf-8")
                        for t in text.split("\n\n"):
                            t = t.strip()
                            if t.startswith("data:"):
                                t = t.split("data:")[1].strip()
                            if not t or t == "[DONE]":
                                continue
                            yield_response.append(t)
                            try:
                                yield json.loads("".join(yield_response))
                                yield_response = []
                            except json.JSONDecodeError:
                                pass
                                # waiting full chunk
                finally:
                    # Close the request before exiting session context. Important to do it here
                    # as if stream is not fully exhausted, we need to close the request nevertheless.
                    await result.aclose()

            return wrap_resp(), got_stream, self.api_key
        else:
            # Close the request before exiting session context.
            return resp, got_stream, self.api_key

    def handle_error_response(self, rbody, rcode, resp, rheaders, stream_error=False):
        try:
            error_data = resp["error"]
        except (KeyError, TypeError):
            raise error.APIError(
                "Invalid response object from API: %r (HTTP response code " "was %d)" % (rbody, rcode),
                rbody,
                rcode,
                resp,
            )

        if "internal_message" in error_data:
            error_data["message"] += "\n\n" + error_data["internal_message"]

        util.log_info(
            "OpenAI API error received",
            error_code=error_data.get("code"),
            error_type=error_data.get("type"),
            error_message=error_data.get("message"),
            error_param=error_data.get("param"),
            stream_error=stream_error,
        )

        # Rate limits were previously coded as 400's with code 'rate_limit'
        if rcode == 429:
            return error.RateLimitError(error_data.get("message"), rbody, rcode, resp, rheaders)
        elif rcode in [400, 404, 415]:
            return error.InvalidRequestError(
                error_data.get("message"),
                error_data.get("param"),
                error_data.get("code"),
                rbody,
                rcode,
                resp,
                rheaders,
            )
        elif rcode == 401:
            return error.AuthenticationError(error_data.get("message"), rbody, rcode, resp, rheaders)
        elif rcode == 403:
            return error.PermissionError(error_data.get("message"), rbody, rcode, resp, rheaders)
        elif rcode == 409:
            return error.TryAgain(error_data.get("message"), rbody, rcode, resp, rheaders)
        elif stream_error:
            # TODO: we will soon attach status codes to stream errors
            parts = [error_data.get("message"), "(Error occurred while streaming.)"]
            message = " ".join([p for p in parts if p is not None])
            return error.APIError(message, rbody, rcode, resp, rheaders)
        else:
            return error.APIError(
                f"{error_data.get('message')} {rbody} {rcode} {resp} {rheaders}",
                rbody,
                rcode,
                resp,
                rheaders,
            )

    def request_headers(self, method: str, extra, request_id: Optional[str]) -> Dict[str, str]:
        user_agent = "OpenAI/v1 PythonBindings/%s" % (version.VERSION,)
        if openai.app_info:
            user_agent += " " + self.format_app_info(openai.app_info)

        uname_without_node = " ".join(v for k, v in platform.uname()._asdict().items() if k != "node")
        ua = {
            "bindings_version": version.VERSION,
            "httplib": "requests",
            "lang": "python",
            "lang_version": platform.python_version(),
            "platform": platform.platform(),
            "publisher": "openai",
            "uname": uname_without_node,
        }
        if openai.app_info:
            ua["application"] = openai.app_info

        headers = {
            "X-OpenAI-Client-User-Agent": json.dumps(ua),
            "User-Agent": user_agent,
        }

        headers.update(util.api_key_to_header(self.api_type, self.api_key))

        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        if self.api_version is not None and self.api_type == ApiType.OPEN_AI:
            headers["OpenAI-Version"] = self.api_version
        if request_id is not None:
            headers["X-Request-Id"] = request_id
        if openai.debug:
            headers["OpenAI-Debug"] = "true"
        headers.update(extra)

        return headers

    def _validate_headers(self, supplied_headers: Optional[Dict[str, str]]) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if supplied_headers is None:
            return headers

        if not isinstance(supplied_headers, dict):
            raise TypeError("Headers must be a dictionary")

        for k, v in supplied_headers.items():
            if not isinstance(k, str):
                raise TypeError("Header keys must be strings")
            if not isinstance(v, str):
                raise TypeError("Header values must be strings")
            headers[k] = v

        # NOTE: It is possible to do more validation of the headers, but a request could always
        # be made to the API manually with invalid headers, so we need to handle them server side.

        return headers

    def _prepare_request_raw(
        self,
        url,
        supplied_headers,
        method,
        params,
        files,
        request_id: Optional[str],
    ) -> Tuple[str, Dict[str, str], Optional[bytes]]:
        abs_url = "%s%s" % (self.api_base, url)
        headers = self._validate_headers(supplied_headers)

        data = None
        if method == "get" or method == "delete":
            if params:
                encoded_params = urlencode([(k, v) for k, v in params.items() if v is not None])
                abs_url = _build_api_url(abs_url, encoded_params)
        elif method in {"post", "put"}:
            if params and files:
                data = params
            if params and not files:
                data = json.dumps(params).encode()
                headers["Content-Type"] = "application/json"
        else:
            raise error.APIConnectionError(
                "Unrecognized HTTP method %r. This may indicate a bug in the "
                "OpenAI bindings. Please contact us through our help center at help.openai.com for "
                "assistance." % (method,)
            )

        headers = self.request_headers(method, headers, request_id)

        util.log_debug("Request to OpenAI API", method=method, path=abs_url)
        util.log_debug("Post details", data=data, api_version=self.api_version)

        return abs_url, headers, data

    def request_raw(
        self,
        method,
        url,
        *,
        params=None,
        supplied_headers: Optional[Dict[str, str]] = None,
        files=None,
        stream: bool = False,
        request_id: Optional[str] = None,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = None,
    ) -> Union[httpx.Response, Generator[str, None, None]]:
        abs_url, headers, data = self._prepare_request_raw(url, supplied_headers, method, params, files, request_id)

        try:
            sync_method: Callable = self.session_stream_sync if stream else self.sync_session.request

            if files:
                data, content_type = requests.models.RequestEncodingMixin._encode_files(files, data)  # type: ignore
                headers["Content-Type"] = content_type

            result = sync_method(
                method,
                abs_url,
                headers=headers,
                data=data,
                content=files,
                params=params,
                timeout=request_timeout if request_timeout else TIMEOUT_SECS,
            )
            if not stream:
                util.log_debug(
                    "OpenAI API response",
                    path=abs_url,
                    response_code=result.status_code,
                    processing_ms=result.headers.get("OpenAI-Processing-Ms"),
                    request_id=result.headers.get("X-Request-Id"),
                )

                # Don't read the whole stream for debug logging unless necessary.
                if openai.log == "debug":
                    util.log_debug("API response body", body=result.content, headers=result.headers)

        except httpx.TimeoutException as e:
            raise error.Timeout("Request timed out: {}".format(e)) from e
        except httpx.RequestError as e:
            raise error.APIConnectionError("Error communicating with OpenAI: {}".format(e)) from e

        return result

    async def arequest_raw(
        self,
        method,
        url,
        session: httpx.AsyncClient,
        *,
        params=None,
        supplied_headers: Optional[Dict[str, str]] = None,
        files=None,
        stream: bool = False,
        request_id: Optional[str] = None,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = None,
    ) -> httpx.Response:
        abs_url, headers, data = self._prepare_request_raw(url, supplied_headers, method, params, files, request_id)

        timeout = request_timeout if request_timeout else TIMEOUT_SECS

        if files:
            data, content_type = requests.models.RequestEncodingMixin._encode_files(files, data)
            headers["Content-Type"] = content_type

        request_kwargs = {
            "method": method,
            "url": abs_url,
            "headers": headers,
            "data": data,
            "params": params,
            "files": files,
            "timeout": timeout,
        }

        try:
            async_method: Callable = self.session_stream_async if stream else self.async_session.request
            result = await async_method(**request_kwargs)
            result.raise_for_status()  # This will raise an exception for HTTP errors.
            util.log_info(
                "OpenAI API response",
                path=abs_url,
                response_code=result.status_code,
                processing_ms=result.headers.get("OpenAI-Processing-Ms"),
                request_id=result.headers.get("X-Request-Id"),
            )
            # Don't read the whole stream for debug logging unless necessary.
            if openai.log == "debug":
                util.log_debug("API response body", body=result.content, headers=result.headers)
            return result
        except httpx.TimeoutException as e:
            raise error.Timeout("Request timed out") from e
        except httpx.RequestError as e:
            raise error.APIConnectionError("Error communicating with OpenAI") from e

    def _interpret_response(
        self, result: Union[httpx.Response, Any], stream: bool
    ) -> Tuple[Union[OpenAIResponse, Iterator[OpenAIResponse]], bool]:
        """Returns the response(s) and a bool indicating whether it is a stream."""
        if stream:
            res = []
            text = result.read().decode("utf-8")
            for t in text.split("\n\n"):
                final_t = t.split("data:")[1].strip()
                if final_t == "[DONE]":
                    break
                res.append(
                    self._interpret_response_line(
                        final_t,
                        result.status_code,
                        result.headers,
                        stream=False,
                    )
                )
            res = (r for r in res)
            return res, True
        else:
            return (
                self._interpret_response_line(
                    result.text,
                    result.status_code,
                    result.headers,
                    stream=False,
                ),
                False,
            )

    def _interpret_response_line(self, rbody: str, rcode: int, rheaders, stream: bool) -> OpenAIResponse:
        # HTTP 204 response code does not have any content in the body.
        if rcode == 204:
            return OpenAIResponse(None, rheaders)

        if rcode == 503:
            raise error.ServiceUnavailableError(
                "The server is overloaded or not ready yet.",
                rbody,
                rcode,
                headers=rheaders,
            )
        try:
            if "text/plain" in rheaders.get("Content-Type", ""):
                data = rbody
            else:
                data = json.loads(rbody)
        except (JSONDecodeError, UnicodeDecodeError) as e:
            raise error.APIError(f"HTTP code {rcode} from API ({rbody})", rbody, rcode, headers=rheaders) from e
        resp = OpenAIResponse(data, rheaders)
        # In the future, we might add a "status" parameter to errors
        # to better handle the "error while streaming" case.
        stream_error = stream and "error" in resp.data
        if stream_error or not 200 <= rcode < 300:
            raise self.handle_error_response(rbody, rcode, resp.data, rheaders, stream_error=stream_error)
        return resp

    def session_stream_sync(self, *args, **kwargs) -> Union[httpx.Response, Coroutine[Any, Any, httpx.Response]]:
        request = self.sync_session.build_request(*args, **kwargs)
        response = self.sync_session.send(
            request=request,
            auth=self.sync_session.auth,
            follow_redirects=self.sync_session.follow_redirects,
            stream=True,
        )
        return response

    async def session_stream_async(self, *args, **kwargs):
        request = self.async_session.build_request(*args, **kwargs)
        response = await self.async_session.send(
            request=request,
            auth=self.async_session.auth,
            follow_redirects=self.async_session.follow_redirects,
            stream=True,
        )
        return response
