import asyncio
import openai
import sys
import traceback
import os
import inspect
from functools import wraps
import termcolor
import dotenv
import threading
from concurrent.futures import ThreadPoolExecutor


from .dependencies import setup_logger
logger = setup_logger(__name__)


dotenv.load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def gpt_error(func):
    if inspect.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                logger.error("getting diagnostic from GPT-4, please wait...")
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, handle_error(e, func, exc_type, exc_value, exc_traceback))
                    future.add_done_callback(lambda x: logger.error(x.result()))
                raise e


        return async_wrapper
    else:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                logger.error("getting diagnostic from GPT-4, please wait...")
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, handle_error(e, func, exc_type, exc_value, exc_traceback))
                    future.add_done_callback(lambda x: logger.error(x.result()))
                raise e

            
        return sync_wrapper

async def handle_error(e, func, exc_type, exc_value, exc_traceback):
    # exc_type, exc_value, exc_traceback = sys.exc_info()
    stack_trace = traceback.extract_tb(exc_traceback)
    last_trace = stack_trace[-1]
    filename = last_trace[0]
    line_number = last_trace[1]
    func_name = last_trace[2]

    # Get and print the source code of the function
    func_code = inspect.getsource(func)

    error_message = f"This error message occurred because of '{str(e)}' at line {line_number} in file {filename}, function {func_name}. The exception type is {exc_type}. The function is: \n\n```python\n{func_code}\n```\n\n Provide a detailed explanation on why this error occurred and sample python code on how to rectify the problem."

    # Make an API call to OpenAI chat with the error message
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Think this through step by step."},
            {"role": "system", "content": "Respond in a terse manner."},
            {"role": "user", "content": error_message}
        ]
    )
    
    return completion.choices[0].message.content