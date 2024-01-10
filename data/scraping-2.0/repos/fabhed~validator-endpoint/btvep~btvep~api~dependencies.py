import concurrent.futures
import json
import logging
from datetime import datetime
from math import ceil
from typing import Annotated
import uuid
import jwt

import openai
import redis.asyncio
import rich
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, OAuth2PasswordBearer
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

from btvep.config import Config
from btvep.constants import COST
from btvep.db.api_keys import ApiKey
from btvep.db.api_keys import get_by_key as get_api_key_by_key
from btvep.db.request import Request as DBRequest
from btvep.db.user import User
from btvep.db.utils import db, db_state_default

config = Config().load()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
token_auth_scheme = HTTPBearer()


from starlette.requests import Request
from starlette.responses import Response


async def reset_db_state():
    db._state._state.set(db_state_default.copy())
    db._state.reset()


def get_db(db_state=Depends(reset_db_state)):
    try:
        db.connect()
        yield
    finally:
        if not db.is_closed():
            db.close()


async def InitializeRateLimiting():
    try:
        redis_instance = redis.asyncio.from_url(
            config.redis_url, encoding="utf-8", decode_responses=True
        )

        async def rate_limit_identifier(request: Request):
            return request.headers.get("Authorization").split(" ")[1]

        await FastAPILimiter.init(redis_instance, identifier=rate_limit_identifier)
    except redis.asyncio.ConnectionError as e:
        rich.print(
            f"[red]ERROR:[/red] Could not connect to redis on [cyan]{config.redis_url}[/cyan]\n [red]Redis is required for rate limiting.[/red]"
        )
        raise e


filter = None
if config.openai_filter_enabled:
    if config.openai_api_key is None:
        raise Exception("OpenAI filter enabled, but openai_api_key is not set.")
    from btvep.filter import OpenAIFilter

    filter = OpenAIFilter(config.openai_api_key)


async def authenticate_user(token: str = Depends(oauth2_scheme)):
    config = Config().load()

    # This gets the JWKS from a given URL and does processing so you can
    # use any of the keys available
    jwks_url = f"https://{config.auth0_domain}/.well-known/jwks.json"
    jwks_client = jwt.PyJWKClient(jwks_url)

    signing_key = None
    try:
        signing_key = jwks_client.get_signing_key_from_jwt(token).key
    except jwt.exceptions.PyJWKClientError as error:
        print("jwks error", error)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error.__str__(),
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.exceptions.DecodeError as error:
        print("jwt decode error", error)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error.__str__(),
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = jwt.decode(
            token,
            signing_key,
            algorithms="RS256",
            audience=config.auth0_api_audience,
            issuer=config.auth0_issuer,
        )
    except Exception as e:
        print("token decode error", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        user, created = User.get_or_create(id=payload["sub"])
        if created:
            print("created user", user)
        return user
    except Exception as e:
        print("user creation error", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


async def authenticate_admin(user=Depends(authenticate_user)):
    if not user.is_admin == 1:
        raise HTTPException(status_code=403, detail="User is not an admin")
    return user


async def authenticate_api_key(
    request: Request, token: str = Depends(token_auth_scheme)
) -> ApiKey:
    input_api_key = token.credentials
    print("authenticating api key", input_api_key)

    def raiseKeyError(detail: str):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )

    async def createErrorRequest(error: str):
        api_request_id = str(uuid.uuid4())
        DBRequest.create(
            is_api_success=False,
            api_request_id=api_request_id,
            api_error=error,
            prompt=json.dumps((await request.json())["messages"]),
            api_key=input_api_key,
        )

    if (input_api_key is None) or (input_api_key == ""):
        createErrorRequest("APIKeyMissing")
        raiseKeyError("Missing API key")

    api_key = get_api_key_by_key(input_api_key)
    if api_key is None:
        createErrorRequest("APIKeyInvalid")
        raiseKeyError("Invalid API key")

    elif api_key.enabled == 0:
        createErrorRequest("APIKeyDisabled")
        raiseKeyError("API key is disabled")
    elif (api_key.valid_until != -1) and (
        api_key.valid_until < datetime.now().timestamp()
    ):
        createErrorRequest("APIKeyExpired")
        raiseKeyError(
            "API key has expired as of "
            + str(datetime.utcfromtimestamp(api_key.valid_until))
        )
    elif not api_key.has_unlimited_credits() and api_key.credits - COST < 0:
        createErrorRequest("APIKeyNotEnoughCredits")
        raiseKeyError("Not enough credits")

    ###  API key is now validated. ###

    if filter:
        messages = (await request.json())["messages"]
        messageContents = [message["content"] for message in messages]
        try:
            check_res = filter.safe_check(messageContents)
            if check_res["any_flagged"]:
                createErrorRequest("FlaggedByOpenAIModerationFilter")
                raiseKeyError("OpenAI moderation filter triggered")
        except concurrent.futures.TimeoutError as e:
            logging.warning("OpenAI filter timed out. Allowing request.")
            pass
        except openai.error.AuthenticationError:
            logging.warning("OpenAI filter auth error. Allowing request.")
            pass

    return api_key


def get_rate_limits(api_key: str = None) -> list[RateLimiter]:
    """Get rate limits. Leave api_key as None to get global rate limits."""

    if not config.rate_limiting_enabled:
        return []
    rate_limits = None
    if api_key is not None:
        rate_limits = json.loads(api_key.rate_limits)
    elif config.rate_limiting_enabled:
        rate_limits = config.global_rate_limits

    HTTP_429_TOO_MANY_REQUESTS = 429

    async def ratelimit_callback(request, response, pexpire, limit):
        print(
            f"Rate limit triggered for ratelimit rule: {limit}",
        )
        api_request_id = str(uuid.uuid4())
        DBRequest.create(
            is_api_success=False,
            api_request_id=api_request_id,
            api_error="RateLimitExceeded",
            prompt=json.dumps((await request.json())["messages"]),
            api_key=request.headers.get("authorization").split(" ")[1],
        )

        expire = ceil(pexpire / 1000)

        raise HTTPException(
            HTTP_429_TOO_MANY_REQUESTS,
            "Too Many Requests",
            headers={"Retry-After": str(expire)},
        )

    rate_limiters = [
        RateLimiter(
            times=limit["times"],
            seconds=limit["seconds"],
            callback=lambda request, response, pexpire: ratelimit_callback(
                request, response, pexpire, limit
            ),
        )
        for limit in rate_limits
    ]

    # Sort the list in ascending order by the ratio of milliseconds to times
    sorted_rate_limiters = sorted(
        rate_limiters, key=lambda rl: rl.milliseconds / rl.times
    )

    return sorted_rate_limiters


global_rate_limits = get_rate_limits()


def VerifyAPIKeyAndLimit():
    async def a(
        request: Request,
        response: Response,
        api_key: Annotated[ApiKey, Depends(authenticate_api_key)],
    ):
        for ratelimit in (
            get_rate_limits(api_key)
            if api_key and api_key.rate_limits
            else global_rate_limits
        ):
            await ratelimit(request, response)

    return a
