import os
import openai
import json
from dotenv import load_dotenv


class Copilot:
    GET_IN_UZBEK = " provide answer in Uzbek language"


    def clear_text(self, text):
        a = text.replace("\n", " ")
        b = a.split()
        c = " ".join(b)

        return c


    def get_answer(self, question):
        prompt = question + self.GET_IN_UZBEK
        
        load_dotenv()

        openai.api_key = os.getenv("CHAT_GPT3_API_KEY1")
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=512,
            temperature=0.5,
        )

        json_object = response

        # Convert the JSON object to a JSON string
        json_string = json.dumps(json_object)

        # Parse the JSON string using json.loads()
        parsed_json = json.loads(json_string)

        text = parsed_json['choices'][0]['text']
        cleared_text = self.clear_text(text)
        
        return cleared_text


# Usage:
# copilot = Copilot()
# a = copilot.get_answer("Birinchi Jahon urishini kim boshlagan?")

# print(a)

# Usage:
# tts = TextToSpeech()
# tts.to_speech(a)
