# train_model_wrapper.py
# author: Diego Magdaleno
# This script acts as a wrapper to the model training process set up by n-sheppard's
# repository. Refer to https://github.com/nshepperd/gpt-2/issues for any issues
# regarding training the GPT-2 model.
# Python 3.7
# Windows/MacOS/Linux


import os
import sys
import json
import time
import string
import platform
import subprocess
from datetime import datetime
from shutil import copyfile, rmtree
from distutils.dir_util import copy_tree


# Determine the version of python that is running and figure out which python command
# to use for python 3.
# @param: Takes no arguments.
# @return: Returns the python command string for the system.
def get_python_version():
    if not sys.version_info.major == 3 and sys.version_info.minor >= 6:
        print("Error: Requires Python 3.6 or higher.")
        exit(1)

    # Send the following command with the subprocess command to determine the path
    # variable set up for python.
    python_command = ""
    command = subprocess.Popen("python3 -V", shell=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
    command_output, command_error = command.communicate()
    if len(command_error) != 0:
        python_command = "python"
    else:
        python_command = "python3"

    # Return the python command string.
    return python_command


# Load in the text from AO3. Clean it as necessary and return the text with the
# special delimiter at the end.
# @param: file_name, the name of the text to be read from AO3.
# @return: Returns the cleaned text from the specified file.
def read_AO3_text(file_name):
    # Read in the file.
    file = open("./AO3/" + file_name, "r", encoding="utf-8")
    file_lines = file.readlines()
    file.close()

    # Return an empty string if the text title is not in English (usually a 
    # good indicator that the rest of the text is not in English.
    bad_title_char = 0
    for char in file_name:
        if char not in string.printable:
            bad_title_char += 1
    if bad_title_char // len(file_name) >= 0.5:
        return ""

    # Return an empty string if the text is not in English.
    bad_line_count = 0
    for line in file_lines:
        bad_char_count = 0
        for char in line:
            if char not in string.printable:
                bad_char_count += 1
        if bad_char_count // len(line) >= 0.9:
            bad_line_count += 1
    if bad_line_count // len(file_lines) >= 0.9:
        return ""
    
    # Strip out the text generated at the beginning and end of every AO3 text.
    start_text = "Summary\n"
    end_text = "Afterword\n"
    #end_text = "Chapter End Notes\n"
    start_index = -1
    end_index = -1
    for line_index in range(len(file_lines)):
        if start_text == file_lines[line_index]:
            start_index = line_index
        elif end_text == file_lines[line_index]:
            end_index = line_index

    # Return the resulting text with the special delimiter appended at the end.
    #delimiter = "\n<|endoftext|>\n"
    #return "\n".join(file_lines[start_index + 1:end_index]) + delimiter
    return "\n".join(file_lines[start_index - 2:end_index]) 


# Load in the text from Gutenberg. Clean it as necessary and return the text with the
# special delimiter at the end.
# @param: folder, the subject folder the text is contained in from Gutenberg.
# @param: file_name, the name of the text to be read from Gutenberg.
# @return: Returns the cleaned text from the specified file.
def read_Gutenberg_text(folder, file_name):
    # Read in the file.
    file = open("./Gutenberg/" + folder + "/" + file_name, "r", encoding="utf-8")
    file_lines = file.readlines()
    file.close()

    # Return an empty string if the text is not in English.
    if "Language: English\n" not in file_lines:
        return ""
    
    # Strip out the text generated at the beginning and end of every Project
    # Gutenberg text.
    start_text = "*** START OF THIS PROJECT GUTENBERG EBOOK "
    end_text = "*** END OF THIS PROJECT GUTENBERG EBOOK "
    start_index = -1
    end_index = -1
    for line_index in range(len(file_lines)):
        if start_text in file_lines[line_index]:
            start_index = line_index
        elif end_text in file_lines[line_index]:
            end_index = line_index

    # Return the resulting text with the special delimiter appended at the end.
    #delimiter = "\n<|endoftext|>\n"
    #return "\n".join(file_lines[start_index + 1:end_index]) + delimiter
    return "\n".join(file_lines[start_index + 1:end_index]) 


# Retrieve all the clean text data from AO3 as a string.
# @param: Takes no arguments.
# @return: Returns a compilation of all the text data from AO3.
def load_from_AO3():
    '''
    # Initialize an empty string to contain all the text data from the books stored
    # in the AO3 folder.
    ao3_text = ""
    
    # Retrieve a list of all texts stored in the AO3 folder. Iterate through each
    # one and extract the cleaned text from each title. Add that cleaned text to
    # the compilation string.
    text_files = os.listdir("./AO3/")
    for text in text_files:
        ao3_text += read_AO3_text(text)
                
    # Return the string containing the clean compilation of all text data from AO3.
    return ao3_text
    '''
    # Retrieve a list of all texts stored in the subfolders. Iterate through each
    # one and extract the cleaned text from each title. Save that cleaned text into
    # the training folder.
    text_files = os.listdir("./AO3/")
    for text in text_files:
        clean_text = read_AO3_text(text)
        if clean_text == "":
            continue
        new_file = open("./gpt-2-finetuning/training/" + text, "w+", encoding="utf-8")
        new_file.write(clean_text)
        new_file.close()
    
    # Return True. This is more for of a status to whichever function calls this. All
    # operations in this function have been successfully executed.
    return True


# Retrieve all the clean text data from Project Gutenberg as a string.
# @param: Takes no arguments.
# @return: Returns a compilation of all the text data from Project Gutenberg.
def load_from_Gutenberg():
    '''
    # Initialize an empty string to contain all the text data from the books stored
    # in the Gutenberg folder.
    gutenberg_text = ""

    # Retrieve a list of all subfolders (specific fiction genres) in the project
    # Gutenberg folder. Iterate through each one.
    gutenberg_folders = os.listdir("./Gutenberg/")
    for folder in gutenberg_folders:
        # Retrieve a list of all texts stored in the subfolders. Iterate through each
        # one and extract the cleaned text from each title. Add that cleaned text to
        # the compilation string.
        text_files = os.listdir("./Gutenberg/" + folder + "/")
        for text in text_files:
            gutenberg_text += read_Gutenberg_text(folder, text)
                
    # Return the string containing the clean compilation of all text data from Project
    # Gutenberg.
    return gutenberg_text
    '''
    # Retrieve a list of all subfolders (specific fiction genres) in the project
    # Gutenberg folder. Iterate through each one.
    gutenberg_folders = os.listdir("./Gutenberg/")
    for folder in gutenberg_folders:
        # Retrieve a list of all texts stored in the subfolders. Iterate through each
        # one and extract the cleaned text from each title. Save that cleaned text into
        # the training folder.
        text_files = os.listdir("./Gutenberg/" + folder + "/")
        for text in text_files:
            clean_text = read_Gutenberg_text(folder, text)
            if clean_text == "":
                continue
            new_file = open("./gpt-2-finetuning/training/" + text, "w+", encoding="utf-8")
            new_file.write(clean_text)
            new_file.close()
    
    # Return True. This is more for of a status to whichever function calls this. All
    # operations in this function have been successfully executed.
    return True


# Load all the training text from the sources list to memory, clean it, and save it
# to a file.
# @param: model_name, the unique model name to save the compiled training texts under.
# @param: sources, the list of valid sources that contain the individual training texts.
# @return: Returns a boolean as to whether the text was successfully loaded
def load_text_to_memory(model_name, sources):
    '''
    # Initialize an empty string to contain all the compiled text data from all sources
    # passed in.
    compiled_text = ""

    # Iterate through the list of sources. Extract the compiled text data from each and
    # add that to the compilation string.
    for source in sources:
        if source == "AO3":
            compiled_text += load_from_AO3()
        elif source == "Gutenberg":
            compiled_text += load_from_Gutenberg()

    # Open and write the compilation string to the appropriately named text file.
    training_file = open("./gpt-2-finetuning/training/" + model_name +\
                            "_training_text.txt", "w+", encoding="utf-8")
    training_file.write(compiled_text)
    training_file.close()

    # Return True. This is more for of a status to whichever function calls this. All
    # operations in this function have been successfully executed.
    return True
    '''
    # Iterate through the list of sources. Clean the text data from each and save it to
    # the training folder.
    for source in sources:
        if source == "AO3":
            source_status = load_from_AO3()
            if not source_status:
                print("Error: There was an error in cleaning the texts from AO3.")
                return source_status
        elif source == "Gutenberg":
            source_status = load_from_Gutenberg()
            if not source_status:
                print("Error: There was an error in cleaning the texts from Gutenberg.")
                return source_status

    # Return True. This is more for of a status to whichever function calls this. All
    # operations in this function have been successfully executed.
    return True


def main():
    # Check the version of python running.
    python_command = get_python_version()

    # Check for all the required directories (GPT-2 repository from OpenAI and the
    # GPT-2-finetuning repository from n-sheppard).
    if not os.path.exists("gpt-2-finetuning") or not os.path.exists("gpt-2-finetuning"):
        print("Error: Missing n-sheppard repository. Run setup.py or git clone" +\
                "https://github.com/nshepperd/gpt-2.git to retrieve the repository. " +\
                "If you decide to use the git clone command, be sure to rename the" +\
                " folder to \"gpt-2-finetuning\'.")
        exit(1)
    elif not os.path.exists("gpt-2") or not os.path.exists("gpt-2"):
        print("Error: Missing OpenAI repository. Run setup.py or git clone" +\
                "https://github.com/openai/gpt-2.git to retrieve the repository.")
        exit(1)

    # Check for all the desired directories (AO3 and Gutenberg).
    valid_sources = ["AO3", "Gutenberg"]

    # Have the user enter the model they wish to train.
    models_list = ["1.5B", "1558M", "774M", "355M", "345M", "124M"]
    valid_model_input = False
    model_selected = ""
    while not valid_model_input:
        user_input = input("Enter the model you want to train: ")
        if user_input in models_list:
            model_selected = user_input
            valid_model_input = True
        else:
            print("Entered an invalid model. Valid models include " + ", ".join(models_list))

    # Check the OpenAI GPT-2 repository for a copy of the valid model the user entered.
    # If it doesn't exist, use the download_model.py program in the repository to
    # retrieve it from OpenAI.
    gpt_2_models = os.listdir("./gpt-2/models")
    if model_selected not in gpt_2_models:
        get_model_command = subprocess.Popen(python_command + " ./gpt-2/download_model.py " +\
                                                model_selected, shell=True,
                                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        get_model_output, get_model_error = get_model_command.communicate()
        print(get_model_output.decode("utf-8"))
        if len(get_model_error) != 0:
            print(get_model_error.decode("utf-8"))
            exit(1)
    if not os.path.exists("./gpt-2-finetuning/src/models"):
        os.mkdir("./gpt-2-finetuning/src/models")
    gpt_2_finetune_models = os.listdir("./gpt-2-finetuning/src/models")
    if model_selected not in gpt_2_finetune_models:
        copy_tree("./gpt-2/models/" + model_selected, 
                    "./gpt-2-finetuning/src/models/" + model_selected)

    # Copy the model from the OpenAI repository to the n-sheppard repository under the
    # src/models folder. Rename it based on the date, model, and name given by the user.
    # This will be referred to as the unique model name.
    user_name = ""
    valid_name_input = False
    while not valid_name_input:
        user_name = input("Enter a name for this model: ")
        if user_name != "":
            valid_name_input = True
        else:
            print("Entered an invalid save name for this model.")
    today = datetime.now().strftime("%Y-%m-%d")
    unique_model_name = today + "_" + user_name + "_" + model_selected

    # Iterate through the training data, cleaning and removing and unecessary text as
    # well as inserting the special delimiter. Combine all texts into one. Save the
    # combined texts as unique model name + "_training_text.txt".
    print("Creating training text file(s) (this may take a few minutes)...")
    if not os.path.exists("./gpt-2-finetuning/training"):
        os.mkdir("./gpt-2-finetuning/training")
    status = load_text_to_memory(unique_model_name, valid_sources)
    if not status:
        print("Error: Failed to create training text file.")
        exit(1)

    # Check to see if the necessary files from the n-sheppard repository are within the
    # src folder. Copy them to there if that is not the case.
    required_files = ["encode.py", "train.py"]
    src_contents = os.listdir("./gpt-2-finetuning/src")
    for file_name in required_files:
        if file_name not in src_contents:
            copyfile("./gpt-2-finetuning/" + file_name, "./gpt-2-finetuning/src/" + file_name)

    # Copy the train_models.sh script over to ./gpt-2-finetuning/src/ folder. Then run
    # it to encode the training data and train the model. Note that the train_models.sh
    # script needs to have execute permissions (chmod +x). It is best to run this script
    # from linux but if on a Windows system, use cygwin to execute.
    print("Encoding data and training model (this will definitely take a few hours)...")
    os.chmod("train_models.sh", 777)
    if "train_models.sh" not in os.listdir("./gpt-2-finetuning/src/"):
        copyfile("train_models.sh", "./gpt-2-finetuning/src/train_models.sh")
        os.chmod("./gpt-2-finetuning/src/train_models.sh", 777)
    os.chdir("./gpt-2-finetuning/src/")
    command_string = "train_models.sh " + model_selected + " ../training " + python_command +\
                        " " + user_name
    if platform.system() != "Windows":
        command_string = "./" + command_string
    encode_and_train = subprocess.Popen(command_string, shell=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
    output, error = encode_and_train.communicate()
    print(output.decode("utf-8"))
    if len(error) != 0:
        print(error.decode("utf-8"))

    # Exit the program.
    print("Model training in progress. Exiting program.")
    exit(0)


if __name__ == '__main__':
    main()
