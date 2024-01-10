# ========================================================================
# Copyright 2023 Emory University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================

__author__ = 'Jinho D. Choi'

import random
from enum import Enum
from typing import Dict, Any

import openai
from emora_stdm import DialogueFlow

from src import utils
from src.utils import MacroGPTJSON, MacroNLG

PATH_USER_INFO = 'resources/userinfo.json'


class V(Enum):
    call_names = 0,  # str
    office_location = 1  # str
    office_hours = 2  # dict -> {"Monday": {"begin": "14:00", "end": "15:00"}}


def main() -> DialogueFlow:
    transitions = {
        'state': 'start',
        '`Hi, how should I call you?`': {
            '#SET_CALL_NAMES': {
                '`Nice to meet you,` #GET_CALL_NAME `. Can you tell me where your office is and when your general office hours are?`': {
                    '#SET_OFFICE_LOCATION_HOURS': {
                        '`Can you confirm if the following office infos are correct?` #GET_OFFICE_LOCATION_HOURS': {
                        }
                    }
                }
            },
            'error': {
                '`Sorry, I didn\'t understand you.`': 'end'
            }
        }
    }

    macros = {
        'GET_CALL_NAME': MacroNLG(get_call_name),
        'GET_OFFICE_LOCATION_HOURS': MacroNLG(get_office_location_hours),
        'SET_CALL_NAMES': MacroGPTJSON(
            'How does the speaker want to be called?',
            {V.call_names.name: ["Mike", "Michael"]}),
        'SET_OFFICE_LOCATION_HOURS': MacroGPTJSON(
            'Where is the speaker\'s office and when are the office hours?',
            {V.office_location.name: "White Hall E305", V.office_hours.name: [{"day": "Monday", "begin": "14:00", "end": "15:00"}, {"day": "Friday", "begin": "11:00", "end": "12:30"}]},
            {V.office_location.name: "N/A", V.office_hours.name: []},
            set_office_location_hours
        ),
    }

    df = DialogueFlow('start', end_state='end')
    df.load_transitions(transitions)
    df.add_macros(macros)
    return df


def get_call_name(vars: Dict[str, Any]):
    ls = vars[V.call_names.name]
    return ls[random.randrange(len(ls))]


def get_office_location_hours(vars: Dict[str, Any]):
    return '\n- Location: {}\n- Hours: {}'.format(vars[V.office_location.name], vars[V.office_hours.name])


def set_office_location_hours(vars: Dict[str, Any], user: Dict[str, Any]):
    vars[V.office_location.name] = user[V.office_location.name]
    vars[V.office_hours.name] = {d['day']: [d['begin'], d['end']] for d in user[V.office_hours.name]}


if __name__ == '__main__':
    openai.api_key_path = utils.OPENAI_API_KEY_PATH
    main().run()
