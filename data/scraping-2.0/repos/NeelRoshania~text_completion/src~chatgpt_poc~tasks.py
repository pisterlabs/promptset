import json
import logging
import time

from datetime import datetime
from celery import states
from chatgpt_poc import app, cparser
from chatgpt_poc.funcs import exponential_backoff
from chatgpt_poc.text_completion import request_completion
from celery.app.log import TaskFormatter
from celery.utils.log import get_task_logger
from celery.result import AsyncResult
from celery.signals import task_success, task_retry, after_setup_task_logger
from multiprocessing.dummy import Pool

from openai.error import RateLimitError, InvalidRequestError

# logging configurations
LOGGER = get_task_logger(__name__)

# signals

# define task logger and redirect to file
@after_setup_task_logger.connect
def setup_task_logger(logger, *args, **kwargs):
    LOGGER = logging.getLogger(__name__)
    LOGGER.handlers.clear()
    LOGGER.addHandler(logging.FileHandler(f'logs/{__name__}.log'))
    for handler in LOGGER.handlers:
        handler.setFormatter(TaskFormatter('[%(asctime)s:%(task_id)s:%(task_name)s:%(name)s:%(levelname)s] %(message)s'))
    return None

# handle task successes
@task_success.connect
def log_task_id(sender=None, result=None, **kwargs) -> tuple:
    LOGGER.info(f'task success - sender:{type(sender)} completed')
    return None

# signal to handle task successes
@task_retry.connect
def retry_feedback(sender=None, request=None, reason=None, einfo=None, **kwargs) -> tuple:
    LOGGER.info(f'task retrying - {reason})')
    return None


# tasks

@app.task(bind=True)
def completion_task(self, prompt: str) -> dict:

    """
        A request to gather a text completion response

    """

    # LOGGER.info(f'task.request:{dir(self.request)} - args=({x}, {y})')

    try:
        start_time = datetime.utcnow().isoformat()
        res = request_completion(prompt=prompt)

        if res["success"]:
            return {
                "task_description": "completion_task",
                "result": res["choices-text"],
                "started_at": f'{start_time}',
                "completed_at": f'{datetime.utcnow().isoformat()}'
            } 
        else:
            LOGGER.error(f'task failed - {e}')
            raise Exception('Unknown exception')
    
    except RateLimitError as e:
        LOGGER.error(f'RATE LIMIT RAISED - retries: {self.request.retries}')
        raise self.retry(
            countdown=exponential_backoff(int(cparser["openai"]["ratelimit"]), self.request.retries if self.request.retries is not None else 0), # custom back-off with jitter
            max_retries=int(cparser["openai"]["maxretries"]),
            exc=e
        )

   # except InvalidRequestError as e:
   #     LOGGER.error(f'task failed - {e}')

# non-tasks

def fetch_task_result(taskid: str) -> tuple:
    
    # unable to retrieve retries

    _res = AsyncResult(id=taskid, app=app)
    if _res.state in ['SUCCESS', 'FAILURE']:
        return taskid, _res.state, _res.date_done.isoformat(), _res.result, 0 if _res.retries is None else _res.retries
    else:
        return taskid, _res.state,

def fetch_task_results(task_ids: list) -> list:

    """
        Gather task results asynchronously
            - Tasks that take time to complete should be waited and returned when complete

    """

    with Pool() as pool:
        results = pool.map(fetch_task_result, task_ids)

    return results

def await_tasks_completion(taskids: list) -> None:
    
    """
        Continuously check task status and terminate when there are not more tasks being retried

    """
    LOGGER.info(f'checking status of tasks: {len(taskids)}')
    
    while True:
        res = fetch_task_results(taskids) 
        if len([r[0] for r in res if r[1] in ('RETRY', 'PENDING')]) == 0:
            break

    LOGGER.info(f'all tasks complete')
    return res
