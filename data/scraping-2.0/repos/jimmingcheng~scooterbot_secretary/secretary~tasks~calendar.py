from typing import Any
from typing import Dict
from typing import Optional

import aiohttp
import arrow
import staticconf
from dataclasses import dataclass
from llm_task_handler.handler import OpenAIFunctionTaskHandler
from llm_task_handler.handler import ProgressMessageFunc
from llm_task_handler.handler import TaskState

from secretary.calendar import get_calendar_service
from secretary.database import UserTable


def google_apis_api_key() -> str:
    return staticconf.read('google_apis.api_key', namespace='secretary')  # type: ignore


@dataclass
class AddCalendarEventArgs:
    title: str
    start_time: arrow.Arrow
    end_time: arrow.Arrow
    is_all_day: bool
    location: Optional[str]
    confirmation_message: str


class AddCalendarEvent(OpenAIFunctionTaskHandler):
    def task_type(self) -> str:
        return 'add_calendar_event'

    def intent_selection_function(self) -> dict:
        return {
            'name': 'add_calendar_event',
            'description': f"Current time is {arrow.now().format('YYYY-MM-DDTHH:mm:ssZZ')}. Prepare to add a calendar event",
            'parameters': {
                'type': 'object',
                'properties': {
                    'title': {
                        'type': 'string',
                    },
                    'start_time': {
                        'type': 'string',
                        'description': 'Start time in the format YYYY-MM-DDTHH:mm:ssZZ. "this weekend" = the coming weekend.',
                    },
                    'end_time': {
                        'type': 'string',
                        'description': 'End time in the format YYYY-MM-DDTHH:mm:ssZZ',
                    },
                    'is_all_day': {
                        'type': 'boolean',
                        'description': 'Whether it is an all day event',
                    },
                    'location': {
                        'type': 'string',
                        'description': 'Location of the event',
                    },
                    'confirmation_message': {
                        'type': 'string',
                        'description': """
Human-readable confirmation of the fields that the event has been added with. e.g.:

Added to your calendar:

>>> Title: **Doctor's appointment**
Date/Time: **October 5, 2023, 4-6:30pm**
Location: **123 Main St, San Francisco, CA 94105**
                        """,
                    },
                },
                'required': ['title', 'start_time', 'end_time', 'is_all_day', 'confirmation_message'],
            }
        }

    async def transition(
        self,
        cur_state: TaskState,
        progress_message_func: Optional[ProgressMessageFunc] = None,
    ) -> TaskState:
        args = cur_state.custom_state

        if cur_state.state_id == self.INTENT_SELECTION_STATE_ID:
            return TaskState(
                handler=self,
                user_prompt=cur_state.user_prompt,
                custom_state=AddCalendarEventArgs(
                    title=args['title'],
                    start_time=arrow.get(args['start_time']),
                    end_time=arrow.get(args['end_time']),
                    is_all_day=args['is_all_day'],
                    location=args.get('location'),
                    confirmation_message=args['confirmation_message'],
                )
            )

        else:

            event: Dict[str, Any] = {
                'summary': args.title
            }

            if args.location:
                async with aiohttp.ClientSession() as session:
                    url = f'https://maps.googleapis.com/maps/api/place/textsearch/json?key={google_apis_api_key()}&query={args.location}'

                    async with session.get(url) as resp:
                        resp_data = await resp.json()
                        place = resp_data['results'][0] if resp_data.get('results') else None

                        if place:
                            event['location'] = place['formatted_address']

            if args.is_all_day:
                event['start'] = {'date': args.start_time.format('YYYY-MM-DD')}
                event['end'] = {'date': args.end_time.format('YYYY-MM-DD')}
            else:
                event['start'] = {'dateTime': args.start_time.format('YYYY-MM-DDTHH:mm:ssZZ')}
                event['end'] = {'dateTime': args.end_time.format('YYYY-MM-DDTHH:mm:ssZZ')}

            # aiogoogle doesn't work for some reason
            user = UserTable().get(self.user_id)
            get_calendar_service(user['google_apis_user_id']).events().insert(calendarId='primary', body=event).execute()

            return TaskState(
                handler=self,
                user_prompt=cur_state.user_prompt,
                reply=args.confirmation_message,
                is_done=True,
            )
