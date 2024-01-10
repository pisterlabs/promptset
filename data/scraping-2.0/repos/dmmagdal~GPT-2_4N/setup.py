# setup.py
# author: Diego Magdaleno
# This program set's up the necessary environment for running the GPT-2_4N project,
# including install modules, downloading repos, and running download scripts.
# Python 3.7
# Windows/MacOS/Linux


import os
import sys
import subprocess
from download_text import Gutenberg, AO3


def main():
    # Check for the version of Python. The python must be version 3.6+.
    print("Checking for Python version 3.6 or higher...")
    if not sys.version_info.major == 3 and sys.version_info.minor >= 6:
        print("Error: Requires Python 3.6 or higher.")
        exit(1)

    # Send the following command with the subprocess command to determine the path
    # variable set up for python.
    python_commands = []
    command = subprocess.Popen("python3 -V", shell=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
    command_output, command_error = command.communicate()
    if len(command_error) != 0:
        python_commands = ["python", "pip"]
    else:
        python_commands = ["python3", "pip3"]

    # Install the necessary modules from requirements.txt
    print("Installing required modules...")
    install_command = subprocess.Popen(python_commands[1], shell=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
    install_output, install_error = install_command.communicate()
    print(install_output.decode("utf-8"))
    if len(install_error) != 0:
        print(install_error.decode("utf-8"))

    # Clone the github repository from n-sheppard for training GPT-2. Move the
    # repository into a folder under a different name.
    print("Cloning n-sheppard repo for training GPT-2 to gpt-2-training folder...")
    trainer_command = subprocess.Popen("git clone https://github.com/nshepperd/gpt-2.git",
                                        shell=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
    trainer_output, trainer_error = trainer_command.communicate()
    print(trainer_output.decode("utf-8"))
    if len(trainer_error) != 0:
        print(trainer_error.decode("utf-8"))
        print("Failed to clone n-sheppard repo.")
        exit(1)
    os.rename("gpt-2", "gpt-2-finetuning")

    # Clone the github repository from openAI for the GPT-2 model.
    print("Cloning OpenAI repo for GPT-2 to gpt-2 folder...")
    openai_command = subprocess.Popen("git clone https://github.com/openai/gpt-2.git",
                                        shell=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
    openai_output, openai_error = openai_command.communicate()
    print(openai_output.decode("utf-8"))
    if len(openai_error) != 0:
        print(openai_error.decode("utf-8"))
        print("Failed to clone OpenAI repo.")
        exit(1)

    # Install the necessary GPT-2 models.
    print("Downloading OpenAI GPT-2 models...")
    models_list = ["1.5B", "1558M", "774M", "355M", "345M", "124M"]
    for model in models_list:
        model_install = subprocess.Popen(python_commands[0] + "./gpt-2/download_model.py " + model,
                                            shell=True, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE)
        model_output, model_error = model_install.communicate()
        if len(model_error) != 0:
            print("Error upon downloading " + model + " model:")
            print(model_error.decode("utf-8"))

    # Download the necessary text files using the text downloader module.
    print("Downloading training texts...")
    valid_input = False
    while not valid_input:
        user_input = input("Do you wish to download training texts from Project Gutenberg?[Y/n] ")
        if user_input.upper() == "Y" or user_input.lower() == "yes":
            gut_obj = Gutenberg()
            gut_obj.download_all_fiction()
            valid_input = True
        elif user_input.upper() == "N" or user_input.lower() == "no":
            valid_input=True
        else:
            print("Input " + user_input + " is not a valid response.")
    valid_input = False
    while not valid_input:
        user_input = input("Do you wish to download training texts from ArchiveOfOurOwn(AO3)?[Y/n] ")
        if user_input.upper() == "Y" or user_input.lower() == "yes":
            ao3_obj = AO3()
            ao3_obj.download_from_ao3()
            valid_input = True
        elif user_input.upper() == "N" or user_input.lower() == "no":
            valid_input=True
        else:
            print("Input " + user_input + " is not a valid response.")

    # Notify the user that setup is now complete.
    print("Install is complete. Ready to run GPT-2 Novel Novel Neural Network (GPT-2_4N) Project.")

    # Exit the program.
    exit(0)


if __name__ == '__main__':
    main()
