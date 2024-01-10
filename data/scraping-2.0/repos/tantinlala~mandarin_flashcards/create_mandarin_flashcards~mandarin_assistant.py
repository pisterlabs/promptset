from openai import OpenAI
from dotenv import load_dotenv
import json

class MandarinAssistant:
    ASSISTANT_ID = 'asst_LTFL3I74XQLw93v4thJpbpsX'

    def __init__(self):
        load_dotenv()
        self.openai = OpenAI()
        self.assistant = self.openai.beta.assistants.retrieve(self.ASSISTANT_ID)

    """ This function will convert a block of text into flashcards with the following form:
    {
        flashcards: [
            {
                mandarin: "水果"
                english: "fruit"
                pinyin: "shuǐguǒ"
            },
            {
                mandarin: "面包"
                english: "bread"
                pinyin: "miànbāo"
            },
            {
                mandarin: "茶"
                english: "tea"
                pinyin: "chá"
            }
        ]
    }
    """
    def convert_to_flashcards(self, text):
        thread = self.openai.beta.threads.create(
            messages = [
                {
                    "role": "user",
                    "content": text,
                }
            ]
        )

        run = self.openai.beta.threads.runs.create(
            thread_id = thread.id,
            assistant_id = self.assistant.id,
        )

        while (run.status != "requires_action"):
            run = self.openai.beta.threads.runs.retrieve(run_id = run.id, thread_id = thread.id)

        json_string = run.required_action.submit_tool_outputs.tool_calls[0].function.arguments
        
        self.openai.beta.threads.runs.cancel(run_id = run.id, thread_id = thread.id)

        # Print JSON object with pretty formatting
        print(json.dumps(json.loads(json_string), indent=4))

        # Return JSON object containing flashcards
        return json.loads(json_string)
