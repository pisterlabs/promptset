from typing import List
from app.model import Message
import openai
import os
import json
from app.utils import logger
from datetime import datetime, timedelta

openai.api_key = os.environ.get('OPENAI_API_KEY')


next_step_selection_fn = {
    "name": "take_next_step",
    "description": "Takes the next step for the conversation.",
    "parameters": {
        "type": "object",
        "properties": {
            "step_id": {
                "type": "string",
                "description": "The id of the step to take next from the list of possible steps."
            },
        },
        "required": ["step_id"]
    }
}

class NextStepExtractor:
    
    def _build_steps_str(self, steps):
        result = dict()
        for step in steps:
            result[step['name']] = step['description']
        return json.dumps(result)
    
    def run_select_next_step(self, messages: List[Message], steps: List[dict]):
        
        steps_str = self._build_steps_str(steps)

        system_prompt = f"""Given a conversation between a user and an assistant about booking accommodation, take the next step for the conversation.
        Here is the list of possible steps to take:
        {steps_str}
        """
        messages_input = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            messages_input.append({"role": msg.role, "content": msg.text})
        # messages_input.append("role")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages_input,
            functions=[next_step_selection_fn],
            function_call={"name": "take_next_step"},
            temperature=0., 
            max_tokens=500, 
        )
        fn_parameters = json.loads(response.choices[0].message["function_call"]["arguments"])
        logger.debug(f"take_next_step fn_parameters {fn_parameters}")
        return fn_parameters