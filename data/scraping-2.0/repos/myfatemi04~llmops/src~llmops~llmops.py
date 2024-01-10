import contextvars
import time
from typing import Any

import openai

event_log = contextvars.ContextVar("event_log")
event_log.set([])

def chat_completion(model, messages, tag='untagged', **kwargs) -> Any:
    inputs = {"model": model, "messages": messages, **kwargs}
    start_time = time.time()
    output = openai.ChatCompletion.create(**inputs)
    end_time = time.time()
    event_log.get().append({
        "type": "chat_completion",
        "tag": tag,
        "input": inputs,
        "output": output.to_dict(), # type: ignore
        "start_time": start_time,
        "end_time": end_time,
    })
    return output

def monitored(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            context = contextvars.copy_context()
            # Clear event log in this context
            context.run(event_log.set, [])
            context_events = context.get(event_log)
            start_time = time.time()
            result = context.run(func, *args, **kwargs)
            end_time = time.time()
            event_log.get().append({
                "type": "function",
                "name": name,
                "start_time": start_time,
                "end_time": end_time,
                "events": context_events,
            })
            return result
        
        return wrapper

    return decorator
