# system imports
import datetime
import sys, os
import dotenv
from anthropic import Anthropic

# local imports
#sys.path.append('../../general_utils')
from .filesystem_utils import write_json_file
dotenv.load_dotenv(dotenv.find_dotenv())

def simple_interface():
    client = Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )

    message = client.beta.messages.create(
        model="claude-2.1",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Hello, Claude"}
        ],
        system='' # optional
    )
    
    print(message.content)

class Claude_Wrapper:
    def __init__(self, model_name=None, save_conversation=True):
        self.client = Anthropic(
            # defaults to os.environ.get("ANTHROPIC_API_KEY")
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )

        self.save_conversation = save_conversation
        self.history = []
        self.system_prompt = 'You are a friendly assistant'
        self.model_name = model_name if model_name else "claude-2.1"

        print('LLM model ready')

    def __call__(self, input):
        return self.run_inference(input)

    def set_system_prompt(self, system_text):
        self.system_prompt = system_text
    
    def run_inference(self, user_text):
        # add current text to conversation chain
        self.append_text({"role": "user", "content": user_text})

        print('querying model...')
        output = self.client.beta.messages.create(model = self.model_name,
                                                  max_tokens=1024,
                                                  system = self.system_prompt,
                                                  messages = self.history)

        if output.type == "message": # success
            ret_val = None
            for itm in output.content:
                if itm.type == "text":
                    ret_val = itm.text
                    break

            print(f"response: {ret_val}")
            self.append_text({"role": "assistant", "content": ret_val})
        elif output.type == "error": #error
            ret_val = None
            print(f"response ERROR: {output.error.message}")
        else: # unknown output type
            ret_val = None
            print(f"unkown output type:")
            print(output)

        return ret_val

    def append_text(self, text_to_append):
        self.history.append(text_to_append)
        return self.history

    def shutdown(self):
        if self.save_conversation:
            # Find the last "/" and take only the text after it.
            model_save_name = self.model_name+'.json'
            save_timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            save_loc = f"./{model_save_name}_{save_timestamp}"
            write_json_file(save_loc, self.history)


if __name__ == "__main__":
    kwargs = {
        'model_name': None, #use default
        'save_conversation': True,
    }
    client = Claude_Wrapper(**kwargs)
    prompt = 'Tell me a story about a dog and a bird who become friends'
    client.run_inference(prompt)
    client.shutdown()