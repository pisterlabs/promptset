from typing import Optional

import arrow
from llm_task_handler.handler import OpenAIFunctionTaskHandler
from llm_task_handler.handler import ProgressMessageFunc
from llm_task_handler.handler import TaskState

from secretary.write import add_todo


class AddTodo(OpenAIFunctionTaskHandler):
    def task_type(self) -> str:
        return 'add_todo_reminder_or_task'

    def intent_selection_function(self) -> dict:
        return {
            'name': self.task_type(),
            'description': 'Add a todo, reminder, or task. If there are multiple separate requests, process the most recent one.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'task_name': {
                        'type': 'string',
                    },
                    'due_date': {
                        'type': 'string',
                        'description': f'Current date is {arrow.now().format("YYYY-MM-DD")}',
                    },
                },
                'required': ['task_name'],
            }
        }

    async def transition(
        self,
        cur_state: TaskState,
        progress_message_func: Optional[ProgressMessageFunc] = None,
    ) -> TaskState:
        cs = cur_state.custom_state

        due_date = arrow.get(cs['due_date']) if 'due_date' in cs else arrow.now()

        add_todo(self.user_id, cs['task_name'], due_date)

        reply = f'''
Here's your todo:
>>> **{cs['task_name']}**
{due_date.format('dddd, MMMM D, YYYY')}
'''

        return TaskState(
            handler=self,
            user_prompt=cur_state.user_prompt,
            reply=reply,
            is_done=True,
        )
