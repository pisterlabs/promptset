import openai
import time
from dotenv import dotenv_values

class GptLLM:
    def __init__(self):
        self.env_vars = dotenv_values(".secrets")
        mykey = self.env_vars.get("OPENAI_API_SECRET")
        self.client = openai.OpenAI(api_key=mykey)
        self.thread = self.client.beta.threads.create()

    def get_assistant(self, assistant_id):
        # Use the 'beta.assistants' attribute, not 'Assistant'
        assistant = self.client.beta.assistants.retrieve(assistant_id)
        return assistant
    
    def SendMessage(self, message, assistant_id):
             
        # Add a Message to a Thread
        self.client.beta.threads.messages.create(
            thread_id = self.thread.id,
            role = "user",
            content = message
        )
        
        # Run the assistant on the thread
        run = self.client.beta.threads.runs.create(
            thread_id = self.thread.id,
            assistant_id = assistant_id
        )

        count = 0
        
        while True:
            # Retrieve the run status
            run_status = self.client.beta.threads.runs.retrieve(
                thread_id = self.thread.id,
                run_id=run.id
            )

            # If run is completed, get messages
            if run_status.status == 'completed':
                messages = self.client.beta.threads.messages.list(thread_id = self.thread.id)

                # Loop through messages and print content based on role
                result = []
                for msg in reversed(messages.data):
                    obj = {
                        "role": msg.role,
                        "content": msg.content[0].text.value 
                    }
                    result.append(obj)
                
                return result
            
            elif count < 10:
                count += 1
                time.sleep(5)
            else:
                return

       