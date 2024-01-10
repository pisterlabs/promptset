
# Copyright Matthew Kolbe (2024)

"""
Legal Disclaimer:

The author of this software (hereafter referred to as 'Author') hereby expressly disclaims 
any warranty for this software. The software and any related documentation is provided "as 
is" without warranty of any kind, either express or implied, including, without limitation, 
the implied warranties or merchantability, fitness for a particular purpose, or non-infringement. 
The entire risk arising out of use or performance of the software remains with you.

In no event shall the Author be liable for any damages whatsoever (including, without limitation, 
damages for loss of business profits, business interruption, loss of business information, or 
any other pecuniary loss) arising out of the use of or inability to use this software, even if 
the Author has been advised of the possibility of such damages.

By using this software, you are acknowledging and agreeing to the aforementioned conditions and 
limitations. You also agree that the Author shall not be liable for any irresponsible, inappropriate, 
or costly use of the software. This software is intended for educational or experimental purposes 
only and should be operated by the user with full understanding of the potential risks, including 
but not limited to financial loss, data corruption, or system crashes.

USE AT YOUR OWN RISK.
"""


from openai import OpenAI
import traceback
import sys
import json
import os
import shutil
import time


# This function diagnoses an Exception and passes the relevant information on to 
# OpenAI's GPT model to offer commentary and a solution to the problem. If it has
# a solition, this function will overwrite the problem file and store a copy of the 
# old version
def OAIExceptionHandler(exception, client = None, gpt_model = "gpt-4", max_file_len=5000, debug = False):
    print(exception)
    tb = traceback.extract_tb(sys.exc_info()[2])

    if client == None:
        client = OpenAI()
    
    which_file_completion = client.chat.completions.create(
    model=gpt_model,
    messages=[
        {"role": "system", 
         "content": '''You receive data about a stack trace, and return the filepath of the file  
that the I likely made the error in. Do not write anything other than the file path or else I will die.'''},
        {"role": "user", "content": str(tb)}
    ]
    )

    recommended_fname = which_file_completion.choices[0].message.content

    if debug:
        print(f"DEBUG: OpenAI's response to the stack trace location inquery: {recommended_fname}")

    for i, frame in enumerate(reversed(tb)):  # Reverse the traceback
        fname, lineno, func_name, text = frame  # Get the file name from the frame
        if fname == recommended_fname:  # Check if the file exists
            break  # Exit the loop after finding the first existing file
    else:  # If no break occurred in the loop
        print("OpenAI cannot recommend a fix because no file in the stack trace exists to edit.")
        sys.exit()

    print(fname)

    with open(fname, 'r') as file:
        file_contents = file.read()

    exception_info = {
        "file_name": fname,
        "line_number": lineno,
        "function_name": func_name,
        "code": text,
        "error_message": str(exception)
    }

    if len(file_contents) < max_file_len:
        exception_info["module_src"] = file_contents
    else:
        print(f"The candidate error file is too large and costly to submit to OpenAI. File len = {len(file_contents)}")
        print(exception_info)
        sys.exit()

    
    completion = client.chat.completions.create(
    model=gpt_model,
    messages=[
        {"role": "system", 
         "content": '''You receive JSON formatted information about an exception in python.
You use world-class expertise to correct the error that caused the exception. 
Correct other errors if you are extremely confident they are errors; if not leave the code alone. 
You respond in JSON in the following schema: 
{ "comment": <string>, "fully_corrected_module_src": <string> }. 
fully_corrected_module_src must be valid Python code. Add code comments to your corrections. 
Do not add a "}" at the end of the code. 
Leave OAIExceptionHandler alone. 
Exclusively write a valid JSON schema answer or else I will die.'''},
        {"role": "user", "content": json.dumps(exception_info)}
    ]
    )

    resp = completion.choices[0].message.content    

    if debug:
        print(f"DEBUG: OpenAI's response to the code fix query: \n\t{resp}")
    

    json_resp = None

    try:
        json_resp = json.loads(resp)
    except:
        print("OpenAI failed to give back raw JSON. Checking marckdown wrapped JSON instead.")

    # gpt-4-1106-preview keeps wrapping its answer in markdown code labels for some reason. Strip them, then
    # check if the first json_resp didn't load.
    if json_resp == None:
        resp = resp.strip('\"').split('\n', 1)[1].rsplit('\n', 1)[0]
        try:
            json_resp = json.loads(resp)
        except:
            print("OpenAI failed to give back JSON. This was the respose, so no code changes were made:")
            print(completion.choices[0].message.content)
            sys.exit()

    brief_fname = fname[:-3]

    if 'fully_corrected_module_src' in json_resp:
        new_file = f'{brief_fname}{time.time()}.py'
        shutil.copy(fname, new_file)

        print(f"OpenAI has replaced the broken module. A backup copy of the old one is at {new_file}")
        with open(fname, 'w') as file:
            if 'comment' in json_resp:
                file.write("# " + json_resp['comment'] + "\n\n")
            file.write(json_resp['fully_corrected_module_src'])
    else:
        print("OpenAI did not give a recommended fix, so no code changes were made.")
    
    if 'comment' in json_resp:
        print(json_resp['comment'])

    sys.exit()