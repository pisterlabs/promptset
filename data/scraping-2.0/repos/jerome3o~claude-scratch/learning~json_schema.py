import json

from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from pydantic import BaseModel


class CalendarEvent(BaseModel):
    name: str
    description: str
    date: str
    location: str


def main():
    client = Anthropic()

    schema_text = json.dumps(CalendarEvent.model_json_schema(), indent=2)

    completion = client.completions.create(
        model="claude-2",
        max_tokens_to_sample=300,
        prompt=f"{HUMAN_PROMPT} Generate a calendar event for a dentist appointment on Tuesday that follows the schema:\n{schema_text}{AI_PROMPT} {{",
    )
    print(completion.completion)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    main()
