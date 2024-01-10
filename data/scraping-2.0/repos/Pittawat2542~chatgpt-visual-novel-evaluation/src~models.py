import random
import time

import openai

from src.obj_models import Story, Criterion
from src.file_utils import save_raw_output_to_file
from src.config import MODEL, TEMPERATURE


def get_chat_response(prompt: str, model=MODEL, temperature=TEMPERATURE) -> (str, str, float):
    print("Initiated chat with OpenAI API.")
    try:
        completion = openai.ChatCompletion.create(model=model,
                                                  temperature=temperature,
                                                  messages=[
                                                      {"role": "user", "content": prompt}])
        print("Completed chat with OpenAI API.")
        return completion.choices[0].message.content, model, temperature
    except Exception as e:
        print(f"Error: {e}")
        print("Retrying....")
        return get_chat_response(prompt, model, temperature)


def evaluate(criterion: Criterion, story: Story):
    concepts = "Concepts:\ngreenhouse effect, ozone depletion, CO2 emissions, sea level rise, climate change, ice melting, air pollution" if criterion.name == "inspiration" else ""

    prompt = f"""Evaluate the following visual novel game story according to the specified criteria and assign a score with a total of 10 for each criterion, where 10 is the best and 0 is the worst. Provide reasons for your scores. Make sure to output it in a MarkDown code block, i.e., between ```json and ```.

Output format:
```json
{{
"story_id": {story.id},
"{criterion.name}": [{{ "<factor_name>": <int score out of 10>, "reason": <reason for the given score> }}]
}}
```

Criterion description:
{criterion.criterion}{concepts}

Story:
{story.story}"""

    response, model, temp = get_chat_response(prompt, temperature=0)
    save_raw_output_to_file(response, model, temp, story, criterion)


def rate_limit_sleeper():
    sleep_time = random.randint(3, 7)
    print(f"Sleeping for {sleep_time} seconds.")
    time.sleep(sleep_time)
