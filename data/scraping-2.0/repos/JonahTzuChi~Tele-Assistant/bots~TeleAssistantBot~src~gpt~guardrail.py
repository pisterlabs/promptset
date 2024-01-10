import config as cfg

import os
import re
import json

from openai import OpenAI
from openai.resources.moderations import Moderations

class UserGuardRail:
    backend = OpenAI(api_key=cfg.openai_api_key)
    moderator = Moderations(backend)

    @classmethod
    def check(cls, prompt):
        # level-1
        print("level-1")
        result = cls.moderator.create(input=prompt, model="text-moderation-latest")
        is_not_appropriate = any(map(lambda x: x.flagged, result.results))
        if is_not_appropriate:
            return False
        print("level-2")
        # level-2
        prompt_object = {"objective": cfg.assistant_objective, "prompt": prompt}
        completion = cls.backend.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {
                    "role": "system",
                    "content": cfg.l2_moderator_prompt,
                },
                {"role": "user", "content": json.dumps(prompt_object)},
            ],
            max_tokens=512,
            n=1,
            temperature=0,
        )
        response = completion.choices[0].message.content
        print(response)
        try:
            response = json.loads(response)
            score = response["score"]
            return float(score) < 0.5, response['response']
        except:
            print(f"Failed to parse response: {response}")
            return True, 'Failed to parse response'
        

class AssistantGuardRail:
    backend = OpenAI(api_key=cfg.openai_api_key)
    moderator = Moderations(backend)

    @classmethod
    def check(cls, prompt):
        # level-1
        print("level-1: ", prompt)
        result = cls.moderator.create(input=prompt, model="text-moderation-latest")
        is_appropriate = not any(map(lambda x: x.flagged, result.results))
        return is_appropriate
