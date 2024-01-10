import openai

from app.core.config import get_settings
from app.models import Condition
from app.models.exercises import Exercise, Writing, Reading

openai.api_key = get_settings().openai_api_key
openai.organization = get_settings().openai_org

TEMPERATURE = 0.3
MAX_TOKENS = 300
MODEL = "gpt-3.5-turbo-16k"
WRITING_SYS_CONT = (
    "You are a English teacher. Check my answers. Do not repeat my answer."
    "Explain the mistakes and suggest how the sentence can be improved."
)

READING_SYS_CONT = (
    "You are a English teacher. I read and retell the text. Check my answers. Do not repeat my answer."
    "Explain the mistakes and suggest how the sentence can be improved."
)

WRITING_INIT_MSG = (
    'Question:\n {exercise}'
    'Conditions:\n {conditions}'
)

READING_INIT_MSG = (
    'Here is the story:\n {exercise}'
    'Conditions:\n {conditions}'
)


async def make_query_message(
        exercise: Exercise,
        conditions: list[Condition | None],
        answer: str
):
    """
    Make query message for openai.
    """
    conditions_str = '\n'.join([c.text for c in conditions if c])  # convert all conditions to string
    init_msg = ''  # init message for openai
    text = ''  # text from exercise
    sys_cont = ''  # system message for openai
    if isinstance(exercise, Writing):
        init_msg = WRITING_INIT_MSG
        text = exercise.question
        sys_cont = WRITING_SYS_CONT
    elif isinstance(exercise, Reading):
        init_msg = READING_INIT_MSG
        text = exercise.text
        sys_cont = READING_SYS_CONT
    query_message: list = [
        {"role": "system", "content": sys_cont},
        {"role": "user", "content": init_msg.format(
            exercise=text,
            conditions=conditions_str
        )},
        {"role": "user", "content": "Check my answer: " + answer},
    ]
    return query_message


async def get_exam_from_openai(
        exercise: Exercise,
        conditions: list[Condition | None],
        answer: str
):
    """
    Get exam from openai.
    """
    query_message = await make_query_message(exercise, conditions, answer)
    try:
        response = await openai.ChatCompletion.acreate(
            model=MODEL,
            messages=query_message,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
    except openai.error.OpenAIError as e:
        print(f'Error: {e}')
        return "Sorry, I am a little busy now. Try again later."  # TODO: add variable with this message
    out_t, in_t, _ = response.usage.values()
    exam = response.choices[0].message.content
    print(f'Out tokens: {out_t}, In tokens: {in_t}')
    return exam, out_t + in_t
