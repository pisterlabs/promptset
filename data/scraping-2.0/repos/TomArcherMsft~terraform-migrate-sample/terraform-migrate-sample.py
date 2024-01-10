# Build instructions:
# pyinstaller --onefile --add-data ./prompt-inputs:./prompt-inputs migrate-terraform-sample.py

import sys
import os
from pathlib import Path
import keyboard
import time
import re
import argparse
from colorama import Fore, Back, Style
import json
from enum import Enum
import shutil
import requests
import azure.core.exceptions
import openai
from azure.identity import AzureCliCredential

# Azure OpenAI settings
OPENAI_API_BASE                 = 'https://openai-content-selfserv.openai.azure.com/'
OPENAI_VERSION                  = '2023-07-01-preview' # This may change in the future.
OPENAI_API_TYPE                 = 'azure_ad'
OPENAI_ENGINE                   = 'gpt-4-32k-moreExpensivePerToken'

# App constants
PROMPT_INPUT_FILE_NAME          = 'prompt-inputs/prompt-inputs.json'
DEBUG_PROMPT_FILE_NAME          = 'prompt.json'
DEBUG_COMPLETION_FILE_NAME      = 'completion.txt'
MAX_SAMPLES_TO_PRINT            = 5
OUTPUT_DIRECTORY_NAME           = 'outputs'
TEMP_DIRECTORY_NAME             = 'temp'
TEST_RECORD_FILE_NAME           = 'TestRecord.md'

# App globals
sample_root_path                = ''
directories_to_process          = []
sample_inputs_source            = []
sample_outputs_source           = []
debug_mode                      = False
output_path                     = ''
temp_path                       = ''

class AppMode(Enum):
    PROCESS_ALL_SAMPLES_WITHOUT_INTERRUPTION    = 1
    CONFIRM_CONTINUE_AFTER_EACH_SAMPLE          = 2

app_mode = AppMode.CONFIRM_CONTINUE_AFTER_EACH_SAMPLE

class PrintDisposition(Enum):
    SUCCESS = 1
    WARNING = 2
    ERROR   = 3
    UI      = 4
    DEBUG   = 5
    STATUS  = 6

def print_message(text = '', disp = PrintDisposition.STATUS, override_indent = False):

    if disp == PrintDisposition.DEBUG and not debug_mode:
        return

    if not override_indent and debug_mode:
        text = "\t" + text

    if disp == PrintDisposition.SUCCESS:
        color = Fore.GREEN
    elif disp == PrintDisposition.WARNING:
        color = Fore.YELLOW
    elif disp == PrintDisposition.ERROR:
        color = Fore.RED
    elif disp == PrintDisposition.UI:
        color = Fore.LIGHTBLUE_EX
    elif disp == PrintDisposition.DEBUG:
        color = Fore.MAGENTA
    else: # disp == PrintDisposition.STATUS
        color = Fore.WHITE

    print(color + text + Style.RESET_ALL, flush=True)

def write_file(file_name, contents):
    try:
        with open(file_name, "w") as f:
            f.write(contents)
    except OSError as error:
        print_message(f"Failed to write file: {error}", PrintDisposition.ERROR)

def write_dictionary_to_file(file_name, dictionary):
    try:
        with open(file_name, "w") as f:
            f.write(json.dumps(dictionary, indent=4))
    except OSError as error:
        print_message(f"Failed to write file: {error}", PrintDisposition.ERROR)

def generate_new_sample(sample_dir):
    print_message(f"\nGenerating new sample...", PrintDisposition.DEBUG, override_indent=True)

    completion = ''

    try:
        messages = []

        # for every item in sample_inputs_source...
        for i in range(len(sample_inputs_source)):
            messages.append({"role": "user", "content": sample_inputs_source[i]})
            messages.append({"role": "assistant", "content": sample_outputs_source[i]})

        sample_source = get_terraform_source_code(sample_dir, include_file_names=False)
        messages.append({"role": "user", "content": sample_source})

        if debug_mode: # Write the prompt to a file.
            curr_sample_temp_path = get_normalized_path(sample_dir, temp_path)
            curr_sample_temp_path = os.path.join(curr_sample_temp_path, DEBUG_PROMPT_FILE_NAME)
            print_message(f"Prompt file path: {curr_sample_temp_path}", PrintDisposition.DEBUG)

            try:
                print_message(f"Creating directory path: {curr_sample_temp_path}", PrintDisposition.DEBUG)
                os.makedirs(os.path.dirname(curr_sample_temp_path),exist_ok=True)

                print_message(f"Writing Azure OpenAI prompt to: {curr_sample_temp_path}...", PrintDisposition.DEBUG)
                write_dictionary_to_file(curr_sample_temp_path, messages)
            except OSError as error:
                raise ValueError(f"Failed to create temp directory. {error}") from error

        special_chars = '\n'
        if debug_mode:
            special_chars = special_chars + '\t'
        print_message(f"{special_chars}Calling OpenAI for: '{sample_dir}'...")
        time.sleep(1)

        response = openai.ChatCompletion.create(engine=OPENAI_ENGINE,
                                                messages=messages,
                                                temperature=0
                                                )
                                                
        if response:
            completion = response['choices'][0]['message']['content']
    except OSError as error:
        print_message(f"Failed to generate new sample. {error}", PrintDisposition.ERROR)

    time.sleep(1)

    if debug_mode: # Write the completion to a file.
        curr_sample_temp_path = get_normalized_path(sample_dir, temp_path)
        curr_sample_temp_path = os.path.join(curr_sample_temp_path, DEBUG_COMPLETION_FILE_NAME)
        print_message(f"Completion file path: {curr_sample_temp_path}", PrintDisposition.DEBUG)

        try:
            print_message(f"Creating directory path: {curr_sample_temp_path}", PrintDisposition.DEBUG)
            os.makedirs(os.path.dirname(curr_sample_temp_path),exist_ok=True)

            print_message(f"Writing Azure OpenAI completion to: {curr_sample_temp_path}...", PrintDisposition.DEBUG)
            write_file(curr_sample_temp_path, completion)
        except OSError as error:
            raise ValueError(f"Failed to create temp directory. {error}") from error
    
    return completion

def get_prompt_input_source():

    print_message()
    print_message("Getting before and after sample directories from settings file...", PrintDisposition.DEBUG)

    bundle_dir = Path(__file__).parent
    print_message(f"Bundle_dir: {bundle_dir}", PrintDisposition.DEBUG)

    prompt_input_file_name = os.path.join(bundle_dir, PROMPT_INPUT_FILE_NAME)

    try:
        # Open the Inputs file.
        print_message(f"Opening prompt inputs file: {prompt_input_file_name}", PrintDisposition.DEBUG)
        with open(prompt_input_file_name) as inputs_file: 
            # Load the JSON inputs file.
            inputs = json.load(inputs_file)
    except OSError as error:
        raise ValueError(f"Failed to open prompt inputs file ({prompt_input_file_name}). {error}") from error

    # There needs to be at least one line (input).
    if 1 > len(inputs):
        raise ValueError('At least one input/output pair must be specified in the inputs file.')

    # For each line in the file (representing a sample directory)...
    for (before, after) in inputs.items():
        before_dir = os.path.join(bundle_dir, before)
        if file_exists(before_dir):
            sample_inputs_source.append(get_terraform_source_code(before_dir, include_file_names=False))
        else:
            raise ValueError(f"[{prompt_input_file_name}] 'Before' directory not found: {before_dir}")

        after_dir = os.path.join(bundle_dir, after)        
        if file_exists(after_dir):
            sample_outputs_source.append(get_terraform_source_code(after_dir, include_file_names=True))
        else:
            raise ValueError(f"[{prompt_input_file_name}] 'After' directory not found: {after_dir}")

def list_to_string(input_list):

    # Initialize an empty string.
    return_string = ""

    # Traverse elements of list...
    for list_element in input_list:

        # Add element to string.
        return_string += list_element

    # Return string.
    return return_string

def get_file_contents(file):
    file_contents = ""

    with open(file, encoding="utf-8") as f:
        file_contents = f.readlines()

    file_contents = list_to_string(file_contents)
    return file_contents

def get_terraform_source_code(dir, include_file_names):
    print_message(f"Getting Terraform source code for: {dir}", PrintDisposition.DEBUG)

    current_sample_source_code = ""

    # For every file in the source directory...
    for file_name in os.listdir(dir):

        if os.path.isfile(os.path.join(dir, file_name)):

            # DO NOT process TestRecord.md file...
            if file_name != TEST_RECORD_FILE_NAME and file_name != '':
                
                # Append source code for the current directory/file
                if include_file_names:
                    current_file_source_code = ("###" 
                    + file_name 
                    + "###" 
                    + "\n" 
                    + get_file_contents(os.path.join(dir, file_name))
                    + "\n" 
                    + file_name 
                    + ":end\n")
                else:
                    current_file_source_code = ("\n" + get_file_contents(os.path.join(dir, file_name)))

                current_sample_source_code += current_file_source_code

    # Return the source code for the specified directory.
    return current_sample_source_code

def file_exists(path):
    return os.path.exists(path)

def parse_args():
    # Configure argParser for user-supplied arguments.

    argParser = argparse.ArgumentParser()

    argParser.add_argument("-s", 
                           "--sample_directory", 
                           help="Name of input sample directory.", 
                           required=True)

    argParser.add_argument("-r", 
                           "--recursive", 
                           action=argparse.BooleanOptionalAction,
                           help="Processes all subdirectories of specified SAMPLE_DIRECTORY'.", 
                           required=False)

    argParser.add_argument("-d", 
                           "--debug", 
                           action=argparse.BooleanOptionalAction,
                           help=argparse.SUPPRESS, 
                           required=False)

    return argParser.parse_args()

def get_normalized_path(sample_dir, output_path):

    # Get the last directory in the sample_root_path.
    # Example: C:\temp\migrate-terraform-sample\batch ==> batch
    relative_stub_root = os.path.basename(os.path.normpath(sample_root_path))

    # Remove sample_root_path from sample_dir to get the sample's relative path.
    # Example: C:\temp\migrate-terraform-sample\batch\basic\sample1\coolio ==> 
    # basic\sample1\coolio
    relative_sample_path = sample_dir.replace(sample_root_path, '')

    # Remove leading slash from relative_sample_path.
    relative_sample_path = relative_sample_path[1:]

    # Join output_path + relative_stub_root.
    # Example: C:\Users\tarcher\source\repos\migrate-terraform-sample\output
    #        + batch
    #        = C:\Users\tarcher\source\repos\migrate-terraform-sample\output\batch
    output_dir = os.path.join(output_path, relative_stub_root)

    # Join output_dir + relative_sample_path to get the final value to return.
    # Example: C:\Users\tarcher\source\repos\migrate-terraform-sample\output\batch
    #        + basic\sample1\coolio
    #        = C:\Users\tarcher\source\repos\migrate-terraform-sample\output\batch\basic\sample1\coolio
    output_dir = os.path.join(output_dir, relative_sample_path)

    return output_dir

def write_new_sample(sample_dir, file_contents):
    # Write the completion string to the appropriate files
    # based on the file markers within the completion.

    # Get the output path for the sample.
    sample_output_path = get_normalized_path(sample_dir, output_path)
    print_message(f"sample_output_path={sample_output_path}", PrintDisposition.DEBUG)

    # Create the directory for the sample.
    print_message(f"Creating directory for sample output: {sample_output_path}", PrintDisposition.DEBUG)
    os.makedirs(sample_output_path, exist_ok = True)

    if file_contents:
        file_names = re.findall(r'###(.*)###', file_contents)

        if file_names:
            for i in range(len(file_names)):
                current_file = file_names[i]

                beg_m = re.search('###'+ current_file + '###', file_contents)
                if beg_m:
                    end_m = re.search(current_file + ':end', file_contents)
                    if end_m:
                        sub = file_contents[(beg_m.span())[1]:(end_m.span())[0]]
                        sub = sub.strip()

                        curr_qfn = os.path.join(sample_output_path, current_file)
                        print_message("Writing file: " + curr_qfn, PrintDisposition.DEBUG)

                        try:
                            # Write the file.
                            with open(curr_qfn, "w") as f:
                                print_message("", PrintDisposition.DEBUG)
                                f.write(sub)
                        except OSError as error:
                            raise ValueError(f"Failed to write file. {error}") from error
                    else:
                        raise ValueError('Failed to find the end of the file name.')
                else:
                    raise ValueError('Failed to find the beginning of the file name.')
        else:
            raise ValueError('Failed to find any file names in the completion.')
    else:
        raise ValueError('Failed to get a valid completion from OpenAI.')

def get_application_path():
    # Get the application path.
    application_path = ''
    if getattr(sys, 'frozen', False):
        application_path = os.path.dirname(sys.executable)
    elif __file__:
        application_path = os.path.dirname(__file__)
    print_message(f"Application path: {application_path}", PrintDisposition.DEBUG)

    return application_path
    
def init_app(args):

    # Set global debugging flag based on command-line arg.        
    if args.debug:
        global debug_mode
        debug_mode = True

    print_message("\nInitializing application...", PrintDisposition.DEBUG, override_indent=True)

    if debug_mode:
        print_message("Debugging enabled.", PrintDisposition.DEBUG)

    # Set the sample root path based on the command-line arg.
    global sample_root_path
    sample_root_path = os.path.abspath(args.sample_directory)
    print_message(f"Sample root path: {sample_root_path}", PrintDisposition.DEBUG)

    if not file_exists(sample_root_path):
        raise ValueError(f"Sample directory not found: {sample_root_path}")

    # Get the application path.
    application_path = get_application_path()

    # Verify that the application path was found.
    if application_path == '':
        raise ValueError('Failed to get application path.')

    # Set the output path based on the application path.
    global output_path
    output_path = os.path.join(application_path, OUTPUT_DIRECTORY_NAME)
    print_message(f"Output path: {output_path}", PrintDisposition.DEBUG)

    # If output path doesn't exist, create it.
    if not os.path.exists(output_path):
        try:
            print_message("Creating output path...", PrintDisposition.DEBUG)
            os.mkdir(output_path)
        except OSError as error:
            raise ValueError(f"Failed to create output directory. {error}") from error
        
    # Set the temp path based on the application path.
    global temp_path
    temp_path = os.path.join(application_path, TEMP_DIRECTORY_NAME)
    print_message(f"Temp path: {temp_path}", PrintDisposition.DEBUG)

    # If temp path doesn't exist, create it.
    if debug_mode and not os.path.exists(temp_path):
        try:
            print_message("Creating temp path for sample...", PrintDisposition.DEBUG)
            os.mkdir(temp_path)
        except OSError as error:
            raise ValueError(f"Failed to create temp directory. {error}") from error

    # Get the directories (samples) to process.
    get_directories_to_process(args)

    print_message("Application initialized.", PrintDisposition.DEBUG, override_indent=True)
    
def get_directories_to_process(args):

    print_message("Getting directories to process...", PrintDisposition.DEBUG)

    global directories_to_process

    # If the specified sample dir (root) exists...
    if file_exists(sample_root_path):

        # Add the root to the list.
        if len([1 for x in list(os.scandir(sample_root_path)) if x.is_file() and ".tf" == (os.path.splitext(x.name)[1].lower())]) > 0:
            directories_to_process.append(sample_root_path)

        # If recursive flag is set...
        if args.recursive:

            # For every directory in the sample dir...
            for root, dirs, files in os.walk(sample_root_path):

                # For every directory in the sample dir...
                for dir in dirs:

                    # Add the directory to the list.
                    if len([1 for x in list(os.scandir(os.path.join(root, dir))) if x.is_file() and ".tf" == (os.path.splitext(x.name)[1].lower())]) > 0:
                        directories_to_process.append(os.path.abspath(os.path.join(root, dir)))
    else:
        raise ValueError(f"Sample directory not found: {sample_root_path}")

def confirm_plan(args):
    print_message("\nPrinting and confirming the plan...", PrintDisposition.DEBUG, override_indent=True)

    if 0 == len(directories_to_process):
        print_message(f"There are no files to process in the specified directory: '{sample_root_path}'" + (" (including its subdirectories)" if args.recursive else "") + ".", PrintDisposition.UI)
    else:
        print_message()
        print_message("***IMPORTANT***: The Skilling org pays for the use of the Azure OpenAI service based on the number of tokens in the request & response for each generated sample.", PrintDisposition.WARNING)
        print_message()
        print_message("See the pricing article for more information: https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/", PrintDisposition.WARNING)
        print_message()
        print_message("Please review the plan below and reach out for guidance if you think the number of samples might be costly.", PrintDisposition.WARNING)
        print_message()
        print_message("Migration Plan:", PrintDisposition.UI)
        print_message()

        # Print the number of directories to process.
        print_message(f"Number of directories to process (max {MAX_SAMPLES_TO_PRINT} shown): {len(directories_to_process)}", PrintDisposition.UI)
        for i in range(len(directories_to_process)): 
            if i < MAX_SAMPLES_TO_PRINT:
                print_message(f"\t{i+1}: {directories_to_process[i]}", PrintDisposition.UI)
            else:
                break
        
        # If there are more than MAX_SAMPLES_TO_PRINT directories to process...
        if len(directories_to_process) > MAX_SAMPLES_TO_PRINT:
            print_message(f"\t{MAX_SAMPLES_TO_PRINT+1}-{len(directories_to_process)}: Not shown for brevity.", PrintDisposition.UI)

        print_message()

        relative_stub_root = os.path.basename(os.path.normpath(sample_root_path))
        print_message(f"The new sample(s) are written to: '{os.path.join(output_path, relative_stub_root)}...'", PrintDisposition.UI)

        print_message()

        if debug_mode:
            print_message(f"The debug files are written to: '{os.path.join(temp_path, relative_stub_root)}...'", PrintDisposition.DEBUG)
            print_message()

    print_message(f"Are you sure you want to continue processing the {len(directories_to_process)} samples?", PrintDisposition.UI)
    print_message("[Y] Yes [No] No (quits the application)", PrintDisposition.UI)

    while True:
        time.sleep(0.3)

        user_response = keyboard.read_key().upper()

        global app_mode
        if user_response == "Y":
            break
        elif user_response == "N":
            raise ValueError("User cancelled the application.")

        time.sleep(0.3)

    print_message("Printed the plan.", PrintDisposition.DEBUG, override_indent=True)

def confirm_continuation_for_current_sample(index, total, sample_dir):
    print_message("\nConfirming continuation for current sample...", PrintDisposition.DEBUG, override_indent=True)

    process_current_sample = True

    if 1 < index:
        print_message()
    print_message(f"Migrate sample directory {index} of {total}: {sample_dir}", PrintDisposition.UI)
    print_message("Are you sure you want to perform this action?", PrintDisposition.UI)
    print_message("[Y] Yes, process this sample [A] Yes to All, [No] Skip this sample, [Q] Quit the application.", PrintDisposition.UI)

    while True:
        time.sleep(0.3)

        user_response = keyboard.read_key().upper()

        global app_mode
        if user_response == "Y":
            process_current_sample = True
            break
        elif user_response == "A":
            app_mode = AppMode.PROCESS_ALL_SAMPLES_WITHOUT_INTERRUPTION
            process_current_sample = True
            break
        elif user_response == "N":
            process_current_sample = False
            break
        elif user_response == "Q":
            raise ValueError("User cancelled the application.")

        time.sleep(0.3)

    return process_current_sample

def delete_previous_sample_dirs(sample_dir):
    # If the sample output path exists, delete it.
    sample_output_path = get_normalized_path(sample_dir, output_path)
    if os.path.exists(sample_output_path):
        print_message(f"Deleting sample output path: {sample_output_path}", PrintDisposition.DEBUG)
        shutil.rmtree(sample_output_path, ignore_errors=True)

    # If the sample temp path exists, delete it.
    sample_temp_path = get_normalized_path(sample_dir, temp_path)
    if os.path.exists(sample_temp_path):
        print_message(f"Deleting sample temp path: {sample_temp_path}", PrintDisposition.DEBUG)
        shutil.rmtree(sample_temp_path, ignore_errors=True)

def init_azure_openai():
    openai.api_base     = OPENAI_API_BASE
    openai.api_version  = OPENAI_VERSION

    try:
        credential = AzureCliCredential()

        # If AzureCliCredential.get_token() fails, it prints its own error message.
        # So, set color to RED just in case before the call.
        print(Fore.RED)
        token = credential.get_token("https://cognitiveservices.azure.com/.default")
    except azure.identity.CredentialUnavailableError as error:
        # Don't send any text in the exception as AzureCliCredential.get_token() 
        # has already printed its own error message.
        raise ValueError(f"") from error
    except azure.core.exceptions.ClientAuthenticationError as error:
        # Don't send any text in the exception as AzureCliCredential.get_token() 
        # has already printed its own error message.
        raise ValueError(f"") from error
    
    openai.api_type     = OPENAI_API_TYPE
    openai.api_key      = token.token

def main():
    try:
        # Get the command-line args (parameters).
        args = parse_args()

        # Initialize Azure OpenAI.
        init_azure_openai()

        # Initialize the application.
        init_app(args)

        # Print the plan to the user so that they know what is going to happen.
        confirm_plan(args)

        # Get the source code for the samples that are being used as the prompt 
        # to illustrate the "before and after" samples to Azure OpenAI.
        get_prompt_input_source()

        # For each directory to process...
        for i, sample_dir in enumerate(directories_to_process):

            if (app_mode == AppMode.PROCESS_ALL_SAMPLES_WITHOUT_INTERRUPTION
            or confirm_continuation_for_current_sample(i+1, len(directories_to_process), sample_dir)):

                # If the sample directories (output & temp) exists, delete them.
                delete_previous_sample_dirs(sample_dir)

                # Generate the new sample and get the Azure OpenAI completion string.
                completion = generate_new_sample(sample_dir)

                # Write the sample file(s).
                write_new_sample(sample_dir, completion)

                # Print success message.
                print_message(f"\nSample successfully migrated: {sample_dir}", PrintDisposition.SUCCESS)

    except ValueError as error:
        print_message(f"\nFailed to migrate sample(s). {error}", PrintDisposition.ERROR)
    except Exception as error:
        print_message(f"\nFailed to migrate sample(s). {error}", PrintDisposition.ERROR)

main()
