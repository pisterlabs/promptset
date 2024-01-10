import openai
from openai import Client
from dotenv import  dotenv_values
import requests
import logging
import os
import re
config = dotenv_values(dotenv_path="./.env")


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
 
"""
this script allows you to:
- define the assistant API in order to define the generated pipeline output
"""

class PDAL_json_generation_template():
    def __init__(self) -> None:
        self.base_prompt = "I want you to generate the pipeline json file based on PDAL(https://pdal.io/en/2.6.0/) latest version of the documentation along with following parameters"    
        self.self_openai_key = config["OPENAI_KEY"]
        self.base_url = "https://api.openai.com/v1"
        self.instructions = """
        You are super helpful in generating the JSON pipeline specifications using PDAL library (https://pdal.io/en/2.6.0/index.html), given the description of the various parameters like : -  writers 
        - readers
        - filters
        - dimensions
        - types 
        etc ... . you should be checking diligently that that generated JSON file should be correct semantically so that the given user has to only enter the required parameters in the template.
        also do fill the final generated pipeline with the parameters defined for the translation by the user in the query.

        Also the most important thing is to be aware of first checking whether the given pipeline tasks is feasible or not (i.e whether there exists the given writers , readers and other parameters in order to run the given tasks) and let them know  that the query is not correct and rephrase again. 
        """        
    
    
    def define_assistant_parameter(self, bot_name, associated_file_path: str , client):
        """
            this is the one time function call to which you will define the properties like:
            - name
            - instruction prompt
            - model for inference
            - category operation (code interpreter, retriever) etc.
            - additional files (pdal documentation inn pdf) in order provide reference:
                - there can be other details 
        """
        
        file_id = openai.files.create(
            file=open(associated_file_path, "rb"),
            purpose='assistants'
        )
        print('file uploaded as' + str(file_id.id))
        
        self.assistant = client.beta.assistants.create(
            name= bot_name,
            instructions= self.instructions,
            file_ids= [str(file_id.id)],
            tools= [
                {"type": "code_interpreter"}
            ],
            model="gpt-4-1106-preview"
        )
        
    
    def creating_message_thread(self, command_description, client):
        """
        start thread in order to take the user specification and then generate the corresponding pipeline file.
        """

        try:
            message = self.assistant.beta.threads.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=command_description
            )
            thread = client.beta.threads.create(
                messages=message
            )
            
            run_model = client.beta.threads.runs.create(
            thread_id= thread.id,
            assistant_id= self.assistant.id)

        except Exception as e:
            print("In the creating_message exception : " + str(e))
        return run_model, thread

    def check_status(self,run_operation, thread, client:openai.OpenAI):
        try:
            run: openai.beta.threads = client.beta.threads.retrieve(thread_id=thread.id, run_id=run.id)
            if run.status == "completed":
                messages = client.beta.threads.messages.list(thread_id=thread.id)
                print("messages: ")
                for message in messages:
                    assert message.content[0].type == "text"
                    print({"role": message.role, "message": message.content[0].text.value})

                client.beta.assistants.delete(self.assistant.id)
                print("job is generated , now downloading the generated json spec")
            else:
                print("still in process, check after some time")
        except Exception as e:
            print(" exception in check_status function : " + str(e))
        
        pattern_file = re.search("file-\w+", message.content[0])
        if pattern_file:
            fileid = pattern_file.group(1)

        ## downloading the file to the local container.
        openai.files.content(
            file_id=fileid
        )
        
        

        
        
        
# if __name__ == "__main__":
#     print("key is : {}".format(os.getenv("OPENAI_API_KEY")))
#     client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#     _object = PDAL_json_generation_template()
#     _object.define_assistant_parameter(bot_name="toto", associated_file_path="./pdal-latest.pdf", client=client)
#     _object.creating_message_thread(
#         "I want you to generate the pipeline JSON file to convert the demo.laz present in current directory to demo.las in the ./output/ directory",
#         client=client
#         )   