from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.memory import ChatMessageHistory
from typing import Tuple

import utils.prompts as prompts
from utils.workout import WorkoutSpecification, Workout, ALL_EXERCISES

def user_characteristics(user):
    '''
    returns the characteristics (i.e. permanent features) of the
    given user in a dictionary format, as follows:
    {
    "height": int (height in inches),
    "weight": int (weight in lbs),
    "experience_level": int (0 for beginner, 1 for intermediate, 2 for expert),
    "objective": str (),
    "days_per_week": int (from 1 to 7),
    "blacklist": list[str] (list of exercise names that are unacceptable),
    "workout_split": str (),
    }
    '''
    return {
        "height": 71,
        "weight": 135,
        "experience_level": "beginner",
        "objective": "build muscle",
        "days_per_week": 3,
        "blacklist": ["deadlift"],
        "workout_split": "push, pull, leg"
    }


def query_for_workout_specifications(api_key_path: str, history: ChatMessageHistory, sleep_hours: int, query:str)\
    -> Tuple[bool, str | WorkoutSpecification, ChatMessageHistory]:
    with open(api_key_path) as f:
        OPENAI_API_KEY = f.read()

    parser = PydanticOutputParser(pydantic_object=WorkoutSpecification)

    fixing_parser = OutputFixingParser.from_llm(llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY),
                                                parser=parser,
                                                prompt="A field was missing. Please provide all fields!",
                                                max_retries=2)

    if len(history.messages) == 0:
        prompt = prompts.SYSTEM_PROMPT_FOR_INITIAL_USER_WORKOUT_QUERY if sleep_hours >= 0 \
            else prompts.SYSTEM_PROMPT_FOR_INITIAL_USER_WORKOUT_QUERY_WITHOUT_SLEEP
        history.add_message(prompt.format(format_instructions=parser.get_format_instructions()))
    if sleep_hours == -1:
        query += " I slept 8 hours last night."
    history.add_user_message(query)

    chat_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

    response = chat_model(history.messages)
    print(response.content)
    history.add_ai_message(response.content)

    if response.content[0] == "{":
        return True, fixing_parser.parse(response.content), history
    else:
        return False, response.content, history

def query_workout(api_key_path: str, workout_spec: WorkoutSpecification, user, sleep_hours: int) -> Tuple[Workout, ChatMessageHistory]:
    with open(api_key_path) as f:
        OPENAI_API_KEY = f.read()

    parser = PydanticOutputParser(pydantic_object=Workout)

    fixing_parser = OutputFixingParser.from_llm(llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY),
                                                parser=parser,
                                                prompt="A field was missing. Please provide all fields!",
                                                max_retries=2)

    workout_preferences_prompt = prompts.WORKOUT_PREFERENCES_PROMPT if sleep_hours >= 0 \
        else prompts.WORKOUT_PREFERENCES_PROMPT_WITHOUT_SLEEP

    messages = [
        prompts.INITIAL_SYSTEM_PROMPT.format(all_exercises=ALL_EXERCISES,
                                             format_instructions=parser.get_format_instructions()),
        prompts.USER_CHARACTERISTICS_PROMPT.format(**user_characteristics(user)),
        workout_preferences_prompt.format(
            duration=workout_spec.duration,
            intensity_level=workout_spec.intensity_level,
            body_area=workout_spec.bodyarea,
            hours_slept = workout_spec.hours_slept if sleep_hours == -1 else sleep_hours
        ),
    ]

    chat_prompt_template = ChatPromptTemplate.from_messages(messages)

    history = ChatMessageHistory()
    history.add_message(messages[0])
    history.add_message(messages[1])
    history.add_message(messages[2])


    chat_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    response = chat_model(chat_prompt_template.format_messages())
    history.add_message(response)

    return fixing_parser.parse(response.content), history

def query_further(api_key_path: str, prompt: str, history: ChatMessageHistory, user) -> Tuple[Workout, ChatMessageHistory]:
    with open(api_key_path) as f:
        OPENAI_API_KEY = f.read()

    parser = PydanticOutputParser(pydantic_object=Workout)
    fixing_parser = OutputFixingParser.from_llm(llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY),
                                                parser=parser,
                                                prompt="A field was missing. Please provide all fields!",
                                                max_retries=2)

    history.add_message(prompts.FURTHER_QUERY_SYSTEM_CONTEXT.format(format_instructions=parser.get_format_instructions()))
    history.add_user_message(prompt)

    chat_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    response = chat_model(history.messages)

    history.add_message(response)

    return fixing_parser.parse(response.content), history

success, response, history = query_for_workout_specifications("api-key.txt", ChatMessageHistory(), False)
while not success:
    print(response)
    success, response, history = query_for_workout_specifications("api-key.txt", history, False)

workout, history = query_workout("api-key.txt", response, None, False)
print(workout)

while True:
    workout, history = query_further("api-key.txt", input(), history, None)
    print(workout)