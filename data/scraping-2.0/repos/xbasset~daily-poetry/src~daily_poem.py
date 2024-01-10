import openai
import jinja2

import os

PROMPT_EMAIL = """This poem is going to be delivered by email to the user's email address: {{email}}.
The goal is to make this poem as personal as possible. The person reading the poem must be willing to share it with their friends and family.
It's a daily poem, so it will help the user develop meaningful conversations.
The poem should be delivered at {{time}}
You will use the following PERSONALIZATION_INFORMATIONS shared by the user itself: 
####
{{personalization}}
####

Write the poem in the same language as the PERSONALIZATION_INFORMATIONS previously. 
The poem must be: {{style}}.
Think quietly step by step.
No talk, just the poem.
"""


# a class to store the poem craft informations
class DailyPoem:
    def __init__(self, email, time_of_day, personalization, style):
        self.email = email
        self.time_of_day = time_of_day
        self.personalization = personalization
        self.style = style
        self.content = self._write_poem()


    def __str__(self):
        return "Email: " + self.email + " - Date: " + self.time_of_day + " - Personalization: " + self.personalization + " - Style: " + self.style + "\n" + self.content

    def _write_poem(self):
        # use openai GPT-4 to write the poem
        
        # load the template
        template = jinja2.Template(PROMPT_EMAIL)
        # render the template
        prompt = template.render(email=self.email, time=self.time_of_day, personalization=self.personalization, style=self.style)

        # call the openai api
        poem = self._call_chat_openai(prompt, "")["content"]

        return poem
    
        
    def _call_chat_openai(self, system_prompt, user_prompt):
        openai.api_key = os.getenv("OPENAI_API_KEY")

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[

                  {
                      "role": "system",
                      "content": "Your task is to write a poem to the user."
                  },
                {
                      "role": "user",
                      "content": user_prompt
                  }

            ],
            temperature=1,
            max_tokens=512,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message
    
