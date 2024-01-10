import random

from emora_stdm import DialogueFlow
import macros
import spacy
import time
import requests
import json
import os
import sqlite3
import openai
from re import Pattern
from utils import MacroGPTJSON, MacroNLG, MacroGPTJSONNLG, gpt_completion
from macros import macros

import re
from emora_stdm import Macro, Ngrams
from typing import Dict, Any, List
from enum import Enum

PATH_API_KEY = '../resources/openai_api.txt'


def api_key(filepath=PATH_API_KEY) -> str:
    fin = open(filepath)
    return fin.readline().strip()


openai.api_key = api_key()

introduction = {
    'state': 'start',
    '`Hi! I\'m Fido. What\'s your name?`': {
        '#SET_CALL_NAME': {
            '`It\'s nice to meet you,`#GET_CALL_NAME`! What\'s the main problem you\'re facing right now?`': {
                '#GET_PROBLEM_RESPONSE': {
                    '#FILLER_RESPONSE': {
                    }
                }
            }
        }
    }
}

# pretreatment, early in treatment, late in treatment

pretreatment = {
    'state': 'user_understanding_of_prob',
    '#FILLER_RESPONSE `Tell me more about it.`': {
        '#GET_PROBLEM_RESPONSE': {
            'state': 'attempted_solutions',
            '#TOUGH_RESPONSE `How have you tried to solve the problem so far, and how did it work?`': {
                '#GET_PROBLEM_RESPONSE': {
                    'state': 'when_problem_not_present',
                    '#FILLER_RESPONSE `When the problem isn’t present (or isn’t bad), what is going on differently?`': {
                        '#GET_PROBLEM_RESPONSE': {
                            'state': 'summarize_reiterate_problem',
                            '`Thank you for sharing. `#GET_SUMMARY` Just want to make sure I understand.`': {
                                '[{yes, yeah, correct, right, yuh, yep, yeap, yup}]': 'early_in_treatment_base',

                                '[{no, nope, not really, not at all, nah, incorrect, not correct, not right}]': {
                                    '`No worries! Can you please tell me what I didn\'t get right, and what I should have understood?`': {
                                        '#GET_PROBLEM_RESPONSE': {}
                                    }
                                },
                                'error': {
                                    '`Sorry, I didn\'t get that. Can you please tell me what I didn\'t get right, and what I should have understood?`': {
                                        '#GET_PROBLEM_RESPONSE': {}
                                    }
                                }
                            }

                        },
                        'error': {
                            '`I\'m sorry I didn\'t get that. I\'ll ask again.`': 'when_problem_not_present'
                        }
                    }
                },
                'error': {
                    '`I\'m sorry I didn\t get that. I\'ll ask again`': 'attempted_solutions'
                }
            }
        },
        'error':{
            '`I\'m sorry I didn\t get that. I\'ll ask again`': 'user_understanding_of_prob'
        }
    },
}

# IF ALL INFORMATION IS GATHERED, PRESENT A SUMMARY OF THE INFORMATION.
# THEN GO TO EARLY IN TREATMENT.

# action, maintenance, and termination

# questions to ask:
# when and how does the problem influence you; and when do you influence it
# what's your idea or theory about what wil help?
# what are some potential roadblocks or challenges that you anticipate in addressing this issue?
# what are some small steps you can take to address this issue?


# maybe we shouldnt be asking directly

# TODO: Add acknowledgement macro
# TODO: Decide on things to trim.
# TODO: Figure out how to create engaging responses.
early_in_treatment = {
    'state': 'early_in_treatment_base',
    '`Great! Let\'s start by understanding your goals and expectations from this therapy session. Make a goal statement starting with "My goal is"`': {
        '#GET_EARLY_RESPONSE': {
            'state': 'user_emotional_state',
            '#FILLER_RESPONSE`How are you feeling right now?`': {
                '#GET_EARLY_RESPONSE': {
                            'state': 'user_support_system',
                            '#FILLER_RESPONSE`It can be helpful to have people you trust to talk to. Who would you say that is for you currently?`': {
                                '#GET_EARLY_RESPONSE': {
                                                                    'state': 'user_finds_anticipated_challenges',
                                                                    '#FILLER_RESPONSE`What are some blockers or challenges that you anticipate in addressing this issue?`': {
                                                                        '#GET_EARLY_RESPONSE': {
                                                                            'state': 'how_problem_influences_user_vice_versa',
                                                                            '#FILLER_RESPONSE`When and how does the problem influence you?`': {
                                                                                '#GET_EARLY_RESPONSE': {
                                                                                    'state': 'get_user_ideas_on_what_will_help',
                                                                                    '#FILLER_RESPONSE`What\'s your ideas or theories about what will help?`': {
                                                                                        '#GET_EARLY_RESPONSE': {
                                                                                            'state': 'early_in_treatment_summary',
                                                                                            '`Ok, I think I have an idea. `#GET_SUGGESTION `Please let me know if I have understood everything correctly.`': {
                                                                                                '[{yes, yeah, correct, right, yuh, yep, yeap, yup}]': {
                                                                                                    '`Great! Let\'s move on to the next step.`': 'post_treatment_base'
                                                                                                },
                                                                                                '[{no, nope, not really, not at all, nah, incorrect, not correct, not right}]': {
                                                                                                    '`No worries! Can you please tell me what I didn\'t get right, and what I should have understood?`': {
                                                                                                        '#GET_EARLY_RESPONSE': {},
                                                                                                    }
                                                                                                },
                                                                                                'error': {
                                                                                                    '`Sorry, I didn\'t get that. If there was something I got wrong let me know, otherwise I\'ll reiterate my suggestion.`': {
                                                                                                        '#GET_EARLY_RESPONSE': {},
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                        },
                                                                                        'error':{
                                                                                            '`I\'m sorry I didn\t get that. I\'ll ask again`': 'get_user_ideas_on_what_will_help'
                                                                                        }
                                                                                    }
                                                                                },
                                                                                'error':{
                                                                                    '`I\'m sorry I didn\t get that. I\'ll ask again`': 'how_problem_influences_user_vice_versa'
                                                                                }
                                                                            }
                                                                        },
                                                                        'error':{
                                                                            '`I\'m sorry I didn\t get that. I\'ll ask again`': 'user_finds_anticipated_challenges'
                                                                        }
                                                                    }
                                                                },
                                                                'error':{
                                                                    '`I\'m sorry I didn\t get that. I\'ll ask again`': 'user_support_system'
                                                                }
                                                            }
                                                        },
                                                        'error':{
                                                            '`I\'m sorry I didn\t get that. I\'ll ask again`': 'user_emotional_state'
                                                        }
                                                    }
                                                },
                                                'error':{
                                                    '`I\'m sorry I didn\t get that. I\'ll ask again`': 'early_in_treatment_base'
                                                }
                                            }
                                        }











# I feel like that this is okay to leave here, in general it should be a good response for most issues that the user is facing.

# for tough responses, respond in the dialog flow and then let gpt handle states, but make sure the response is tailored to the user.
# maybe a macro #GET_GPT_AWKNOWLEDGEMENT that, when mixed with #GET_FILLER_TEXT, will make sure we're not docked points for just straight copying gpt responses.


# evaluation
post_treatment = {
    'state': 'post_treatment_base',
    '`Do you feel that today\'s session has made a positive impact on your situation?`': {
        '[{yes, yeah, correct, right, yuh, yep, yeap, yup, I do}]': {
            '`I\'m glad to hear it! I think if you follow my suggestion, you\'ll be well on your way to solving your problem!`': {
                'error': {
                '`Happy I could help! Have a good day, and feel free to chat again sometime :)`': 'end'

                                }
                            }
                        }
                    },




        '[{no, nope, not really, not at all, nah, incorrect, not correct, not right}]': {
            '`Can you explain why the session didn\'t positively impact your situation?`': {
                '`I\'m sorry to hear that. If you are still having trouble I recommend reaching out to some professional resources. I\'m still a young chat bot and learning to handle these issues. Let me know if you need anything else.`': {
                    'error':{
                        '`Have a good day, I\'m sorry I wasn\'t more helpful. Feel free to talk again!`': 'end'
                    }
                }
            }
        },
        'error': {
            '`Sorry, I didn\'t get that.`': {
                'state': 'post_treatment_base'

                }
            }
        }








# we gotta solo implement preparation/action stages, which is where the actual "therapizing" happens


df = DialogueFlow('start', end_state='end')
df.local_transitions(introduction)
df.local_transitions(pretreatment)
df.local_transitions(early_in_treatment)
df.local_transitions(post_treatment)
df.add_macros(macros)

if __name__ == '__main__':
    df.run(debugging=False)


"""
curl https://api.openai.com/v1/models -H "Authorization: Bearer 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" >openai-models-access-$(date +%Y%m%d-%H%M%S).json
"""