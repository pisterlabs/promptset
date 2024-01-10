from typing import Optional

import json
from abc import ABC
from langchain.chat_models.openai import ChatOpenAI
from langchain.schema import HumanMessage
from llm_task_handler.handler import TaskHandler
from llm_task_handler.handler import OpenAIFunctionTaskHandler
from llm_task_handler.handler import TaskState
from llm_task_handler.handler import ShortcutTaskHandler


class TaskSelector(ABC):
    def __init__(self, task_handlers: list[TaskHandler]) -> None:
        self.task_handlers = task_handlers

    def select_task(self, prompt: str) -> Optional[TaskState]:
        return (
            self._try_select_task_from_shortcut_handlers(prompt) or  # noqa: W504
            self._try_select_task_from_slower_handlers(prompt)
        )

    def _try_select_task_from_shortcut_handlers(self, prompt: str) -> Optional[TaskState]:
        fast_handlers = [p for p in self.task_handlers if isinstance(p, ShortcutTaskHandler)]

        for handler in fast_handlers:
            if handler.intent_matches(prompt):
                return TaskState(
                    handler=handler,
                    user_prompt=prompt,
                    custom_state=None,
                )

        return None

    def _try_select_task_from_slower_handlers(self, prompt: str) -> Optional[TaskState]:
        functions = [
            p.intent_selection_function()
            for p in self.task_handlers
            if isinstance(p, OpenAIFunctionTaskHandler)
        ]

        if not functions:
            return None

        chat_model = ChatOpenAI(  # type: ignore
            model_name='gpt-4',
            temperature=0,
            max_tokens=250,
            model_kwargs={"functions": functions},
        )

        reply = chat_model([
            HumanMessage(content=prompt),
        ])

        func_call = reply.additional_kwargs.get('function_call')
        if func_call:
            handler = self._get_task_handler(func_call['name'])
            openai_func_args = json.loads(func_call['arguments'])
            return TaskState(
                handler=handler,
                user_prompt=prompt,
                custom_state=openai_func_args,
                state_id=OpenAIFunctionTaskHandler.INTENT_SELECTION_STATE_ID,
            )

        return None

    def _get_task_handler(self, task_type: str) -> TaskHandler:
        for handler in self.task_handlers:
            if handler.task_type() == task_type:
                return handler

        raise ValueError(f'No processor found for task_type {task_type}')
