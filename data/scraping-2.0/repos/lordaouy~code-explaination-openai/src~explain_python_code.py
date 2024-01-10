# Usage:
# python ./src/explain_python_code.py -i ../gpt-discord-bot/src/ -o ./output/explain_python_code/text_davinci_003/ -m text-davinci-003
# python ./src/explain_python_code.py -i ../gpt-discord-bot/src/ -o ./output/explain_python_code/code_davinci_002/ -m code-davinci-002

import os
import argparse
import sys
import openai
from dotenv import load_dotenv
from utils import prompt_api, format_output

# Set up Azure OpenAI
load_dotenv()
openai.api_type = "azure"
openai.api_base = "https://tutorial-openai-01-2023.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key = os.getenv("OPENAI_API_KEY")
 
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True, help="Path to a folder")
    parser.add_argument("-o", "--output_path", required=True, help="Path to write the result to")
    parser.add_argument("-m", "--model", required=True, help="Name of the model to be used")
    args = vars(parser.parse_args())
    INPUT_PATH = args['input_path']
    OUTPUT_PATH = args['output_path']
    MODEL =args['model']

    # check for valid folder
    if os.path.exists(INPUT_PATH):
        print('input path: ', INPUT_PATH)
    else: 
        print(INPUT_PATH, ' invalid.')
        sys.exit()

    # create OUTPUT_PATH if not existed
    if not os.path.exists(OUTPUT_PATH):
        try:
            os.mkdir(OUTPUT_PATH)
        except OSError:
            print ("Creation of the directory %s failed" % OUTPUT_PATH)
        else:
            print ("Successfully created the directory %s " % OUTPUT_PATH)
    
    # create prompt
    prompt_postfix = """ 
    ###
    Here's what the above code is doing:
    1.
    """

    # get list of files
    file_list = os.listdir(INPUT_PATH)

    for fn in file_list:
        # read from file
        fname = os.path.join(INPUT_PATH, fn); print(fname)
        f = open(fname, "r")
        code = f.read()

        # build input
        prompt = code +  prompt_postfix

        # Configure hyper-parameters
        engine=MODEL
        prompt=prompt
        temperature=0 # [0, 1]
        max_tokens=500
        top_p=1
        frequency_penalty=0 # [0, 2]
        presence_penalty=0 # [0, 2]
        best_of=1
        stop=["###"]

        # make a request to api
        response = prompt_api(engine, prompt, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, best_of, stop)

        # format output
        output = format_output(prompt_postfix, response)

        # write output to file
        fname_out = os.path.join(OUTPUT_PATH, fn + '.output'); print(fname_out)
        output_file = open(fname_out, "w")
        output_file.write(output)
        output_file.close()