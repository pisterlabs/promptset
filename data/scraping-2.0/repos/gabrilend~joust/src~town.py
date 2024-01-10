import openai
from utilities import stream_print

openai.api_key = "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"

class Town:

    def __init__(self, name, tournament=False, melee=False):
        #self.engine_type = "text-ada-001"
        self.engine_type = "text-davinci-003"
        self.character_count = 0

        self.name = name
        self.tournament = tournament
        self.melee = melee
        self.description=""
            
    def __str__(self):
        return self.name

    def describe_town(self, person_name=""):
        if self.description != "":
            print(self.description)
        else:
            self.generate_and_stream_description()

    def generate_and_stream_description(self):
        
        prompt = "Describe the town square of a medieval city named " + \
                 self.name + "\n"

        response = openai.Completion.create(engine=self.engine_type, \
                                            prompt=prompt, \
                                            max_tokens = 512, \
                                            temperature = 1, \
                                            stream=True)
        collected_events = []
        completion_text = ''
        self.character_count = 0

        for event in response:
            collected_events.append(event)
            event_text = event['choices'][0]['text']
            completion_text += event_text
            self.character_count = stream_print(event_text, \
                                                self.character_count)
        self.description = completion_text
        
        print("")
        prompt += self.description

        prompt += "\n"
        prompt += "There is a jousting tournament in town today. Describe the "
        prompt += "exciting atmosphere and the exciting faces of the festival"
        prompt += " goers.\n\n"

        response = openai.Completion.create(engine=self.engine_type, \
                                        prompt=prompt, \
                                        max_tokens = 256, \
                                        temperature = 1, \
                                        stream=True)
        collected_events = []
        completion_text = ''
        self.character_count = 0

        for event in response:
            collected_events.append(event)
            event_text = event['choices'][0]['text']
            completion_text += event_text
            self.character_count = stream_print(event_text, \
                                                self.character_count)
        return completion_text

