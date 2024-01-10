import openai
import random
from utilities import stream_print

openai.api_key = "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"

class character:

    def __init__(self, name="", character_class="commoner", description=""):
        #self.engine_type = "text-ada-001"
        self.engine_type = "text-davinci-003"
        self.name = name
        self.character_class = character_class
        self.description = description
        self.character_count = 0

        health_roll = [1, 2, 3, 4]
        ac_roll = [13, 14, 15, 16, 17, 18]

        if self.name == "":
            self.name = self.generate_name()
        else:
            self.name = name
        if self.description == "":
            self.generate_and_stream_description()
        else:
            self.description = description
        self.health = random.choice(health_roll) + 1
        self.ac = random.choice(ac_roll)
        self.status = self.generate_status()

    def __str__(self):
        return self.name + ", the " + self.character_class

    def generate_status(self):

        prompt = "Describe a person's with these characteristics in +" \
                 "one sentence. Focus on their health and wellbeing.\n"

        if self.description == "":
            print("Error: please run this function after generating a " + \
                  "description for this character.")

        prompt += self.description + "\n"

        if self.health <= 0:
            status = "close to death"
        elif self.health == 1:
            status = "weakened"
        elif self.health == 2:
            status = "average"
        elif self.health == 3:
            status = "healthy"
        elif self.health == 4:
            status = "vigorous"
        elif self.health == 5:
            status = "strong"
        else:
            status = "undefinable"

        prompt += f"In terms of their health, in this moment " + \
                   "they are feeling {status}\n"

        response = openai.Completion.create(engine=self.engine_type, \
                                            prompt=prompt, \
                                            max_tokens = 256, \
                                            temperature = 0.3)
        self.status = response.choices[0].text.strip()
        return self.status

    def generate_and_stream_description(self):

        prompt = f"Describe a medieval person"
        prompt += f" named {self.name} living in the year 1453.\n\n"

        response = openai.Completion.create(engine=self.engine_type, \
                                            prompt=prompt, \
                                            max_tokens = 256, \
                                            temperature = 1, \
                                            stream=True)
        collected_events = []
        completion_text = ""
        self.character_count = 0

        for event in response:
            collected_events.append(event)
            event_text = event['choices'][0]['text']
            completion_text += event_text
            self.character_count = stream_print(event_text, \
                                                self.character_count)
        self.description = completion_text

    def generate_name(self):

        prompt = "Write the name of a medieval person who is travelling through \
                  town. They are a " + self.character_class + "."

        response = openai.Completion.create(engine=self.engine_type, \
                                            prompt=prompt, \
                                            max_tokens = 16, \
                                            temperature = 1)
        return response.choices[0].text.strip()

    def test_functionality(self):
        print("This is a test of the character generation functionality.")
        print("Please stand by as a character is generated.")
        
        character_test = character()
        character_test.generate_description()
        print("Printing character:")
        print(character_test)
        print("Printing character name:")
        print(character_test.name)
        print("Printing character description:")
        print(character_test.description)
        print("Test complete. Goodbye.")


