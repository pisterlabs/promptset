# Controller for Summarization & Next Steps Generation
import os
import time

from flask import request, jsonify, make_response
# from download_hf_models import BrioSummarizer
from openai_requests import *




# Switching to GPT-3 -- need trim into chewable sequences
gpt_max_length = 2048 #in characters, approximating n_tokens that GPT-3 will register


class NextStepsEndpoint():

    # Generate Summaries for given inputs -- cannot do since no GPU memory locally...
    # def generate_summary(self, input_text: list)-> list:
    #     summary_text = brio.summarize(input_text) #expects a string, not list of strings
    #     return brio.bpf(summary_text) #returns bpf'd output, can swap for below -- a list of results
    #     # return summary_text #returns paragraph-esque summary


    # Main Method -- expects a transcript string as in request_data
    def post(self):

        # Key for Text that needs to get summarized
        request_data = request.get_json() #expect following vars in the request
        input_text = request_data.get("input_text")
        # input_text = sample_txt #TODO: comment out the above and delete this -- used for testing the endpoint locally

        # input_tokens = brio.tokenizer.tokenize(input_text) #get tokens for input (using these as an estimate for GPT-3's token requirements)

        max_model_chars = int(gpt_max_length * 4.5)

        # Check Length, Trim if longer than max context for Model -- estimate of actual token len w heurisitic
        print("Lengths of Input & Model Limit", int(len(input_text)/4.5), gpt_max_length)
        if len(input_text) >= gpt_max_length:
            print("Webpage Content too long for Model, trimming into Sequences")
            input_text = list((input_text[i:i + max_model_chars] for i in range(0, len(input_text), max_model_chars)))
        else:
            input_text = [input_text] #whole transcript fits in context

        summarized_text = [] #to hold the summary strings
        sequence_count = 0 #count of sequences to summarize
        total_time = elapsed_time = time.monotonic()
        print(f"\nGenerating Summaries for {len(input_text)} Sequences")

        for sequence in input_text:
            start_time = time.monotonic()
            # sequence_summaries = self.generate_summary(sequence) #generate summary of content per max token len sequence -- cannot use, no GPU
            sequence_summaries = get_summary_gpt(sequence) #use GPT-3 instead
            summarized_text.append(sequence_summaries)

            # for summary in sequence_summaries:
            #     summarized_text.append(summary) #list of strings per bullet point
            end_time = time.monotonic()
            time_diff = end_time - start_time
            elapsed_time += time_diff #to track total time for all summaries
            sequence_count += 1
            print(f"  Sequence {sequence_count} summarized in {time_diff:.2f}s")
        print(f"Completed Summarization of Doc in: {elapsed_time - total_time:.2f}s")


        # Return Summarized String, if not empty list
        if summarized_text != []:
            summarized_text = "\n".join(summarized_text)
            summarized_text = summarized_text.replace(" ", "")
            # print(type(summarized_text)) #debugging
            # print(summarized_text) #view output, serverside
            next_steps = get_steps_gpt(summarized_text)
            return make_response(next_steps, 200)
        else:
            return make_response("Summary Generation Error", 400)


    # Get Request Handler, mainly to check server's home dir & health of endpoint
    def get(self):
        print("Get Request Successful to Summarize Endpoint Successful")
        #output = f"Path: {os.curdir} & Contents{os.listdir()}"
        output = "Summary Endpoint- requires text to be passed via a JSON 'input_text' key\n"
        return make_response(output)


    # Request Handler -- moved in Logic for different request types from Main File
    def request_handler(self):
        if request.method == "GET":
            resp = self.get()
            return resp
        elif request.method == "POST":
            resp = self.post()
            return resp


