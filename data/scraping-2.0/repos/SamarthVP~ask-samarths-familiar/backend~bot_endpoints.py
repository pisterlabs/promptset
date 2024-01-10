from fastapi import APIRouter
from qa_sys import agent, memory
from openai import error
from pydantic import BaseModel
from fastapi.routing import APIRoute
from typing import Callable
from fastapi import APIRouter, Request, Response, status, HTTPException
import time
from functools import wraps

def rate_limit(max_calls: int, time_window: int):
    def rate_limit_dec(func):
        all_calls = []
        
        @wraps(func)
        async def wrapper(request, *args, **kwargs):
            now = time.time()
            calls_in_window = [call for call in all_calls if call > now - time_window]
            if len(calls_in_window) >= max_calls:
                raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")
            all_calls.append(now)
            return await func(request, *args, **kwargs)
        
        return wrapper
    return rate_limit_dec

class LoggingRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            response: Response = await original_route_handler(request)
            print(f"Responded to {await request.body()} with {response.body}")
            return response

        return custom_route_handler

router = APIRouter(prefix="/api/qa", route_class=LoggingRoute)

class QARequest(BaseModel):
    query: str

@router.post("/")
@rate_limit(5, 60)
async def answer(request: QARequest):
    try:
        response = agent["agent"].run(agent["prompt"].format(query=request.query))

    except (error.RateLimitError, error.Timeout) as e:
        # Ratelimit or openai timeout
        return {"response": "Sorry, I'm feeling a little overloaded, could you ask again in a few minutes?"}
    except (error.InvalidRequestError) as e:
        # Correct behaviour should be clear memory and try again since it is likely a max token error
        # If that doesnt work then they should try to reduce the message size
        try:
            memory["memory"].clear()
            response = agent["agent"].run(agent["prompt"].format(query=request.query))
        except (error.InvalidRequestError) as e:
            return {"response": "Sorry, I was distracted by a squirrel while you were talking, could you ask the question more consisely?"}

    return {"response": response}

