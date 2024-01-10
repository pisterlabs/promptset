from typing import Optional

import arrow
import yaml
from langchain.chat_models.openai import ChatOpenAI
from langchain.schema import HumanMessage
from llm_task_handler.handler import OpenAIFunctionTaskHandler
from llm_task_handler.handler import ProgressMessageFunc
from llm_task_handler.handler import TaskState

from secretary.calendar import get_calendar_service
from secretary.database import UserTable


class AnswerQuestionFromCalendar(OpenAIFunctionTaskHandler):
    def task_type(self) -> str:
        return 'answer_question_from_calendar'

    def intent_selection_function(self) -> dict:
        return {
            'name': self.task_type(),
            'description': f"Current time is {arrow.now().format('YYYY-MM-DDTHH:mm:ssZZ')}. Answer a question",
            'parameters': {
                'type': 'object',
                'properties': {
                    'question': {
                        'type': 'string',
                        'description': 'The question to answer',
                    },
                    'start_time': {
                        'type': 'string',
                        'description': 'Start of the question\'s time range in the format YYYY-MM-DDTHH:mm:ssZZ. "this weekend" = the coming weekend.',
                    },
                    'end_time': {
                        'type': 'string',
                        'description': 'End of the question\'s time range in the format YYYY-MM-DDTHH:mm:ssZZ',
                    },
                },
                'required': ['question'],
            }
        }

    async def transition(
        self,
        cur_state: TaskState,
        progress_message_func: Optional[ProgressMessageFunc] = None,
    ) -> TaskState:
        return TaskState(
            handler=self,
            user_prompt=cur_state.user_prompt,
            reply='Next week, you have a PTA meeting on Tuesday at 6:30pm. You also have a dentist appointment on Thursday at 2:00pm.',
            is_done=True,
        )

        args = cur_state.custom_state

        question = args['question']
        start_time = arrow.get(args['start_time']) if args.get('start_time') else None
        end_time = arrow.get(args['end_time']) if args.get('end_time') else None

        # aiogoogle doesn't work for some reason
        user = UserTable().get(self.user_id)
        events = get_calendar_service(user['google_apis_user_id']).events().list(
            calendarId='primary',
            timeMin=start_time.isoformat() if start_time else None,
            timeMax=end_time.isoformat() if end_time else None,
            singleEvents=True,
            orderBy='startTime',
        ).execute().get('items', [])

        prompt = f'''
# Instructions

Answer the question using only the provided events data.

# Question

{question}

# Events

'''

        prompt += self.events_yaml(events)

        import logging; logging.info(prompt)
        chat_model = ChatOpenAI(  # type: ignore
            model_name='gpt-4-1106-preview',
            temperature=0,
            max_tokens=250,
        )

        reply = chat_model([
            HumanMessage(content=prompt),
        ])

        return TaskState(
            handler=self,
            user_prompt=prompt,
            reply=reply,
            is_done=True,
        )

    def events_yaml(self, events: list[dict]) -> str:
        return yaml.dump([
            {
                'when': self._get_time_phrase(event),
                'where': event.get('location'),
                'what': event['summary'],
                'details': event.get('description'),
            }
            for event in events
        ])

    def _get_time_phrase(self, event: dict) -> str:
        start = self._get_event_time(event['start']).to('US/Pacific')
        end = self._get_event_time(event['end']).to('US/Pacific')

        if 'dateTime' in event['start']:
            if start.date() == end.date():
                if start.hour == end.hour and start.minute == end.minute:
                    return f"on {start.format('YYYY-MM-DD')} at {start.format('h:mm a')}"
                else:
                    return f"on {start.format('YYYY-MM-DD')} from {start.format('h:mm a')} to {end.format('h:mm a')}"
            else:
                return f"from {start.naive} to {end.naive}"

        else:
            if start.date() == end.date():
                return f"on {start.format('YYYY-MM-DD')}"
            else:
                return f"from {start.format('YYYY-MM-DD')} to {end.format('YYYY-MM-DD')}"

    def _get_event_time(self, time_dict: dict) -> arrow.Arrow:
        if 'dateTime' in time_dict:
            return arrow.get(time_dict['dateTime'])
        else:
            return arrow.get(time_dict['date'])
