#!/usr/bin/python3
import boto3
import time
import logging
import json
import os
import pprint
import shutil
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI  # new way of import.

# Function to copy the file to the web directory
# you could put this into the class, or later import this.
def copy_to_web_directory(source_file, destination_dir='/var/www/html'):
    if not destination_dir.endswith('/'):
        destination_dir += '/'
    destination_file = destination_dir + source_file.split('/')[-1]
    shutil.copy(source_file, destination_file)
    print(f'File copied to {destination_file}')

class SummarizeChunkService:
    def __init__(self, mydir="./SummarizeChunkDir", log_file="SummarizeChunk.log"):
        self.dir = mydir
        self.log_file = log_file
        self.sqs = boto3.client('sqs', region_name='us-east-2')
        self.input_queue_url = 'https://sqs.us-east-2.amazonaws.com/635071011057/sqs_summarize_chunk.fifo'
        self.stage_1_outputs = [] 

        # Initialize logging
        logging.basicConfig(filename=self.log_file, level=logging.ERROR)

    def generate_html(self):
        summaries=self.stage_1_outputs
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="UTF-8">
        <meta http-equiv="refresh" content="10">
        <title>Summary Output</title>
        <script>
        function toggleSummary(id) {
            var summary = document.getElementById(id);
            if (summary.style.display === 'none') {
                summary.style.display = 'block';
            } else {
                summary.style.display = 'none';
            }
        }
        </script>
        </head>
        <body>
        <h1>Live Summary</h1>
        """

        for index, summary in enumerate(summaries):
            title = summary['title']
            text = summary['text']
            html_content += f"""
            <p><b onclick="toggleSummary('summary{index}')"
            style="cursor:pointer;">{title}</b></p>
            <p id="summary{index}" style="display:none;">{text}</p>
            """
    
        html_content += """
        </body>
        </html>
        """
        return html_content

    def summarize_text(self, text_chunk):
        # This function is now designed to process a single text chunk
        # Call the summarize_stage_1 function here with a single chunk
        # Assume that the function is already refactored to handle single chunks
        summary = self.summarize_stage_1(text_chunk)['stage_1_outputs']

    def process(self, text_chunk):
        try:
            self.summarize_text(text_chunk)
            # Add code to send the summary to the output queue or final storage
        except Exception as e:
            logging.error("Error processing text chunk", exc_info=True)

    def summarize_stage_1(self,chunks_text):

        print(f'-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-', flush=True)
        print(f'\n\n\nStart time: {datetime.now()}', flush=True)

        # Prompt to get title and summary for each chunk
        map_prompt_template = """Firstly, give the following text an informative title. Then, on a new line, write a 75-100 word summary of the following text. Within the summary section, write the text as if you were the author of the text you are reviewing, not from the third personp perspective. For example 'In this text, the author' do not say, 'the text is about', or 'the txt says':
        {text}

      
        Return your answer in the following format:
        Title | Summary...
        e.g.
        Why Artificial Intelligence is Good | AI can make humans more productive by automating many repetitive processes.
      
        TITLE AND CONCISE SUMMARY:"""

        map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
        # Define the LLMs
        map_llm = ChatOpenAI(temperature=0, model_name = 'gpt-3.5-turbo')
        map_llm_chain = LLMChain(llm = map_llm, prompt = map_prompt)
        # Initialize an empty list to store the outputs
        # Loop through each chunk and process it  In this case chunks_text is just the text.. not a list.
        single_prompt_input = {'text': chunks_text}

        try:
           single_prompt_result = map_llm_chain.apply([single_prompt_input])
           output_string=single_prompt_result[0] #it's in a list, but its one element.

           # Split the string on the '|' character
           split_string = output_string['text'].split(' | ')
           output_dict = {
               'title': split_string[0].strip(),  # Using strip() to remove any leading/trailing whitespaces
               'text': split_string[1].strip()
           }
           print(f"\n\nFINAL single_prompt_result:{output_dict}\n\n")
           self.stage_1_outputs.append(output_dict)

        except Exception as e:
           single_prompt_result = None
           print(f'Error processing chunk: {chunks_text}', flush=True)  # Corrected variable name
           print(e, flush=True)

        # At the end of summarize_stage_1 method
        return {
           'stage_1_outputs': output_dict  # Returning the expected key with the result
        }

    def start(self):
        pp = pprint.PrettyPrinter(indent=4,depth=5)

        print("Summarize Chunks Service started.")
        try:
            while True:
                #print("Checking for messages...")
                messages = self.sqs.receive_message(
                    QueueUrl=self.input_queue_url,
                    MaxNumberOfMessages=1,
                    WaitTimeSeconds=20  # Enable long polling
                )
    
                if 'Messages' in messages:
                    for message in messages['Messages']:
                        print(f"Received message:")
                        pp.pprint(message)
                        body_dict = json.loads(message['Body'])
                        if 'text' in body_dict:
                            chunk_text = body_dict['text']
                            self.process(chunk_text)
                        else:
                            print(f"Message does not contain 'text' key: {message}")
                        self.sqs.delete_message(QueueUrl=self.input_queue_url, ReceiptHandle=message['ReceiptHandle'])

                        pp.pprint(self.stage_1_outputs)
                        html_output = self.generate_html()
                        output_file="summary_output.html"
                        with open(output_file, "w") as file:
                            file.write(html_output)
                            copy_to_web_directory(output_file)

      

                else:
                    print("No messages received.")
                # The sleep is not necessary due to long polling
        except KeyboardInterrupt:
            print("Summarize Chunks Service stopped.")

# Load the API key from an environment variable
api_key = os.environ.get("OPENAI_API_KEY")

# Check if the API key is available
if api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

if __name__ == "__main__":
    summarizeChunkService = SummarizeChunkService()
    summarizeChunkService.start()

