"""
gpt_transcript.py

This module contains the class that utilizes the OpenAI's GPT API call to
generate our lab instructions

Created: 07/06/2023

"""

import openai
import tiktoken
import json
from datetime import date

class TranscriptConversion:
    """Class to convert transcription into lab instructions"""

    def __init__(self, model, secret_key):
        """Constructor - sets up OpenAI API's settings

        Args:
            model      (_type_): OpenAI model type used for conversion
            secret_key (_type_): API keys
        """
        self.secret_key = secret_key
        self.model = model
        self.instr_set = None
        self.transcript = None

        self.gpt_prompt = """The following is a timestamped transcript of a lab. Edit it into a clean and concise procedure instruction that would appear in a lab report. Return it as a JSON object with the fields {"Summary":, "Procedure": {"step", "start_time", "end_time"}}. Each step is its own object, can be more than one step per timestamp. Transcript: """
        # self.gpt_prompt = """The following is a timestamped transcript of a lab. Edit it into a clean and concise procedure instruction that would appear in a lab report. Include "Summary" concisely stating the lab's goals, separate with "Procedure", start with "-" for each step, and indicate which timestamp the step was from in "()". Transcript: """

        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
            self.encoding = tiktoken.encoding_for_model(model)
        except Exception as e:
            print("Error: Model or specified transcript location is invalid")

    def generateInstructions(self, transcript_path, encoding="cl100k_base"):
        """
        Generates Instruction set by applying the model on the
        transcript.
        Will keep generating until a valid JSON is returned or reach a max limit of 5 re-generations.

            Args:
                transcript_path      (_type_): location of transcript
                encoding - optional (string): tiktoken encoder base

            Return:
                instr_set      (json): formatted JSON object with fields {"Summary":, "Procedure": [{"Step", "Start_Time", "End_Time"}]}
        """

        # read in transcript txt file
        with open(transcript_path, "r") as file:
            self.transcript = file.read()

        openai.api_key = self.secret_key
        # count tokens to figure out a good max_tokens value
        encoding = tiktoken.get_encoding(encoding)
        encoding = tiktoken.encoding_for_model(self.model)

        # Call GPT4
        raw_instr = None
        json_instr = None

        validJson = False
        maxCalls = 5
        callCount = 0

        # TODO: discuss with team if we should limit to only 4000 tokens for the input and skip the summary when calls are split up.
        while not validJson and callCount < maxCalls:
            msg = [
                {"role": "system", "content": self.gpt_prompt},
                {"role": "user", "content": self.transcript},
            ]
            raw_output = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=msg,
                temperature=0.2,  # in range (0,2), higher = more creative
                # chatcompletion doesn't need max_tokens parameter
            )
            raw_instr = raw_output.get("choices")[0].get("message").get("content")
            try:  # check valid json with the appropriate fields
                json_instr = json.loads(raw_instr)
                json_instr["Summary"]
                json_instr["Procedure"]
                validJson = True
            except Exception as e:
                print(f"Cannot parse JSON. {e} Trying again")
                callCount += 1
            stop_reason = raw_output.get("choices")[0].get("finish_reason")
            if stop_reason != "stop":
                return {
                    "statusCode": 500,
                    "body": "GPT was stopped early because of "
                    + stop_reason
                    + ". Please try again.",
                }
            
        metadata = {
            "version": "Autolab v1.1.0-alpha",
            "author": "Altum Labs",
            "date-generated": date.today().strftime("%Y-%m-%d"),
            "description": "These generated results are a product of Autolab by Altum Labs. It contains private data and is not for distribution. Unauthorized use of this data for any other purposes is strictly prohibited. ",
        }

        result = {
            "metadata": metadata,
            "summary": json_instr["Summary"],
            "procedure": json_instr["Procedure"],
        }
        return result
