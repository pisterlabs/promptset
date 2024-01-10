from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
import os
import json
import time
import base64
import requests

class Chain():

    def __init__(self):
        load_dotenv()
        self.api_key, self.org_id = os.getenv('OPENAI_API_KEY'), os.getenv('OPENAI_ORG_ID')
        self.client = OpenAI()
        self.images_dir = "images"
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
        self.assistant = "asst_6Ug6p8RqTMMNZaXVAgtKUDnK" #"asst_FB7tR3KmVEoKT0gZfN8gy0S0"
        self.thread = self.client.beta.threads.create()

    def _wait_on_run(self, run, thread):
        while run.status == "queued" or run.status == "in_progress":
            run = self.client.beta.threads.runs.retrieve(
                # TODO: make ID
                thread_id=thread,
                run_id=run.id,
            )
            time.sleep(0.5)
        return run

    def _process_code(self, run_steps):
        processed_code = []
        # Check and print the step details
        for step in run_steps.data:
            code_obj = {}
            # if tool used
            if step.type == 'tool_calls':

                # Extract input and output
                input_value = step.step_details.tool_calls[0].code_interpreter.input
                output_value = step.step_details.tool_calls[0].code_interpreter.outputs

                # place into object
                code_obj["input"] = input_value
                code_obj["output"] = output_value
                code_obj["step_id"] = step.id
                code_obj["thread_id"] = step.thread_id
                code_obj["run_id"] = step.run_id
                code_obj["step_details"] = step.step_details
                code_obj["errors"] = step.last_error

                # append to list
                processed_code.append(code_obj)

        return processed_code
    
    def _process_messages(self, messages):
        # process each message
        processed_messages = []

        for message in messages["data"]:
            # Initialize new message object
            message_obj = {}

            message_obj["step_id"] = message["id"]
            message_obj["run_id"] = message["run_id"]
            message_obj["thread_id"] = message["thread_id"]
            message_obj["role"]  = message["role"]

            # Extract the text value and image file_id (if available)
            for content in message["content"]:
                if content["type"] == "text":
                    message_obj["value"] = content["text"]["value"]
                elif content["type"] == "image_file" and "image_file" in content:
                    message_obj["image_file_file_id"] = content["image_file"]["file_id"]

            processed_messages.append(message_obj)

        return processed_messages

    def _response_link(self, processed_code, processed_messages):
        # iterate through each message
        for message in processed_messages:
            if "image_file_file_id" in message:
                # access png if it exists
                image_file_id = message["image_file_file_id"]
                image_data = self.client.files.content(image_file_id)  # Replace 'image_file_id' with actual method to fetch file
                image_data_bytes = image_data.read()

                # save to images folder
                image_file_path = os.path.join(self.images_dir, f"{image_file_id}.png")
                with open(image_file_path, "wb") as file:
                    file.write(image_data_bytes)

                # add this path as a key
                message["IMAGE"] = image_file_path
        
        for code in processed_code:
            for message in processed_messages:
                if code["step_id"] == message["step_id"]:
                    message["input"] = code["input"]
                else: 
                    message["input"] = None

        return processed_messages

    
         # Function to encode the image
    def _encode_image(self,image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    def initial_chain(self, prompt, dataset, gpt_model):
        path = "app_pages\\data\\"
        new_path = os.path.join(path, dataset)

        # Check if the file exists before trying to open it
        if not os.path.isfile(new_path):
            # If the file doesn't exist, raise a FileNotFoundError
            raise FileNotFoundError(f"File not found: {new_path}")

        file = self.client.files.create(
        file = open(new_path, "rb"),
        purpose='assistants'
        )

        # create prompt #TODO: THIS IS WHERE USER PROMPT GOES
        message = self.client.beta.threads.messages.create(
        thread_id=self.thread.id,
        role="user", 
        # """Do K-means and cluster the supermarket data.
        #  Spending Score is something I assign to the customer based on their defined parameters
        #   like customer behavior and purchasing data. I own the mall and want to understand the 
        #   customers like who can be easily converge [Target Customers] so that the sense can be 
        #   given to marketing team and plan the strategy accordingly.""",#,
        content= prompt,
        file_ids= [file.id] #TODO: this is where custom file goes
        )
        
        # prepare thread for running
        run = self.client.beta.threads.runs.create(
        thread_id=self.thread.id,
        assistant_id= self.assistant,
        model= gpt_model
        #model="gpt-4-1106-preview",
        # instructions="additional instructions",
        # tools=[{"type": "code_interpreter"}, {"type": "retrieval"}]
        )

        # run thread
        run = self._wait_on_run(run, self.thread.id)

        # get run steps
        run_steps = self.client.beta.threads.runs.steps.list(
            thread_id=self.thread.id,
            run_id=run.id
        )  
        
        # process through code interpreter
        processed_code = self._process_code(run_steps)

        # get messages
        messages = self.client.beta.threads.messages.list(thread_id=self.thread.id, order="asc")
        # messages = self.client.beta.threads.messages.list(
        #     thread_id=self.thread.id, order="asc", after=message.id
        # )
        # make into json
        messages = json.loads(messages.model_dump_json())
        
        # process each message
        processed_messages = self._process_messages(messages)

        processed_messages = self._response_link(processed_code, processed_messages)

        return processed_messages, processed_code

    def recreate(self, prompt, dataset, gpt_model):
        
        path = "app_pages\\data\\"
        new_path = os.path.join(path, dataset)

        # Check if the file exists before trying to open it
        if not os.path.isfile(new_path):
            # If the file doesn't exist, raise a FileNotFoundError
            raise FileNotFoundError(f"File not found: {new_path}")

        file = self.client.files.create(
        file = open(new_path, "rb"),
        purpose='assistants'
        )
        
        # Create a message to append to our thread
        message = self.client.beta.threads.messages.create(
            thread_id=self.thread.id, 
            role="user", 
            content=prompt,
            file_ids=[file.id])

        # Execute our run
        run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant,
            model= gpt_model
        )
        
        run = self._wait_on_run(run, self.thread.id)

        # get run steps
        run_steps = self.client.beta.threads.runs.steps.list(
            thread_id=self.thread.id,
            run_id=run.id
        )  

        # process through code interpreter
        processed_code = self._process_code(run_steps)
        # Wait for completion

        # Retrieve all the messages added after our last user message
        messages = self.client.beta.threads.messages.list(
            thread_id=self.thread.id, order="asc", after=message.id
        )

        # make into json
        messages = json.loads(messages.model_dump_json())
        
        # process each message
        processed_messages = self._process_messages(messages)

        processed_messages = self._response_link(processed_code, processed_messages)

        # cleanup
        #response = client.beta.threads.delete("thread_abc123")
        #print(response)

        return processed_messages, processed_code

    def vision(self, image_path):

        image_path = f"images/{image_path}.png"

        # Getting the base64 string
        base64_image = self._encode_image(image_path)

        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "Explain this data visualization in laymen terms. Be specific with the data shown. What does this practically mean about the data? Be concise"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
            }
        ],
        "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        return (response.json())