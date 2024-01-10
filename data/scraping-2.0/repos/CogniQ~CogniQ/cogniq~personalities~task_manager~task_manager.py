from __future__ import annotations
from typing import *
import logging

logger = logging.getLogger(__name__)
import json
import textwrap
import dateutil
import dateutil.parser
from datetime import datetime, timezone
import asyncio

from cogniq.config import APP_URL, OPENAI_CHAT_MODEL, TASK_MANAGER_MAX_SLEEP_TIME
from cogniq.personalities import BasePersonality
from cogniq.openai import system_message, user_message, CogniqOpenAI
from cogniq.slack import CogniqSlack, UserTokenNoneError

from .functions import schedule_future_message_function
from .task_store import TaskStore


class TaskManager(BasePersonality):
    def __init__(self, cslack: CogniqSlack, inference_backend: CogniqOpenAI):
        """
        Initialize the BasePersonality.
        :param cslack: CogniqSlack instance.
        """
        self.cslack = cslack
        self.inference_backend = inference_backend
        self.task_store = TaskStore()

    @property
    def description(self) -> str:
        return "I am a helpful assistant. I have the ability to schedule messages for later sending."

    @property
    def name(self) -> str:
        return "Task Manager"

    async def async_setup(self) -> None:
        await self.task_store.async_setup()
        asyncio.create_task(self.start_task_worker())

    def _parse_arguments(self, arguments: str) -> Dict[str, Any]:
        try:
            result = json.loads(arguments)
            result["when_time"] = self._parse_date(result["when_time"])
            return result
        except Exception as e:
            logger.error(f"JSON parsing failed for function call arguments: {e}: {arguments}")
            raise e

    def _parse_date(self, datestring: str) -> datetime:
        try:
            date = dateutil.parser.parse(datestring)
            date.replace(tzinfo=timezone.utc)
            return date
        except Exception as e:
            logger.error(f"Date parsing failed for datestring: {e}: {datestring}")
            raise e

    async def ask(
        self,
        *,
        q: str,
        message_history: List[Dict[str, str]],
        context: Dict[str, Any],
        stream_callback: Callable[..., None] | None = None,
        reply_ts: str | None = None,
        thread_ts: str | None = None,
    ) -> Dict[str, Any]:
        # bot_id = await self.cslack.openai_history.get_bot_user_id(context=context)
        bot_name = await self.cslack.openai_history.get_bot_name(context=context)
        # if the history is too long, summarize it
        message_history = self.inference_backend.summarizer.ceil_history(message_history)
        message_history = [
            system_message(
                "I don't make assumptions about what values to plug into functions. I ask for clarification if a user request is ambiguous. I only use the functions that I have been provided with. I only use a function if it makes sense to do so."
            ),
        ] + message_history

        tasks_response = await self.inference_backend.async_chat_completion_create(
            messages=message_history,
            model="gpt-4-1106-preview",  # [gpt-4-32k, gpt-4, gpt-3.5-turbo]
            function_call="auto",
            functions=[schedule_future_message_function()],
        )
        logger.info(f"tasks_response: {tasks_response}")
        tasks_message = tasks_response["choices"][0]
        if tasks_message["message"]["content"]:
            answer = tasks_message["message"]["content"]
        else:
            function_call = tasks_message["message"]["function_call"]
            try:
                function_arguments = self._parse_arguments(function_call["arguments"])
            except Exception as e:
                return {
                    "answer": f"I had an error parsing the response from OpenAI. See the logs for details and try again.",
                    "response": tasks_response,
                }
            function_name: str = function_call["name"]

            if function_name == "schedule_future_message":
                future_message: str = function_arguments["future_message"]
                when_time: datetime = function_arguments["when_time"]
                when_time = when_time.replace(tzinfo=timezone.utc)
                confirmation_response: str = function_arguments["confirmation_response"]
                logger.info(f"scheduling future message: {future_message} at {when_time}")

                answer = await self.task_store.enqueue_task(
                    future_message=future_message,
                    when_time=when_time,
                    confirmation_response=confirmation_response,
                    context=context,
                    thread_ts=thread_ts,
                )
            else:
                logger.warning(f"unknown function: {function_name}")
                answer = function_call

        return {"answer": answer, "response": tasks_response}

    async def start_task_worker(self) -> None:
        while True:
            await self.task_store.reset_orphaned_tasks()
            # Get the earliest task
            task: Dict[str, Any] | None = await self.task_store.dequeue_task()

            # If there are no tasks, sleep for some time in an incremental fashion, up to 1 minute
            # Interruptable by a enqueue event
            sleep_time = 0
            if task is None:
                if sleep_time is None:
                    sleep_time = 5
                else:
                    sleep_time = max(5, min(TASK_MANAGER_MAX_SLEEP_TIME, sleep_time * 2))
                await asyncio.sleep(sleep_time)
                continue
            else:
                # Calculate how long to sleep until the task's start time
                sleep_time = min(TASK_MANAGER_MAX_SLEEP_TIME, (task["when_time"] - datetime.now(timezone.utc)).total_seconds())

                if sleep_time > 0:
                    # If the task is in the future, sleep until it's time to start
                    await asyncio.sleep(sleep_time)

                # Now it's time to start the task, so we can lock and dequeue it
                try:
                    # lock the task.
                    # TODO: wrap this in a context manager
                    await self.task_store.lock_task(task["id"])

                    # Execute the task
                    logger.info(f"Executing task: {task['future_message']}")
                    try:
                        await self.cslack.chat_postMessage(
                            channel=task["context"]["channel_id"],
                            text=task["future_message"],
                            thread_ts=task["thread_ts"],
                            context=task["context"],
                        )
                    except Exception as e:
                        logger.error(e)
                        raise e

                    # Delete task from queue after it's done
                    await self.task_store.delete_task(task["id"])
                except Exception as e:
                    logger.error(f"Failed to execute task: {task}, error: {e}")
                    await self.task_store.unlock_task(task["id"])
