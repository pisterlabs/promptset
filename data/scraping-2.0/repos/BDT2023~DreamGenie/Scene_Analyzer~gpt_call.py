import openai
from icecream import ic  # for debugging https://github.com/gruns/icecream
import os
import random as rand
import re
import pandas as pd

openai.api_key = os.environ.get('API_KEY_OPENAI')
TEST = True

def load_dreams(file_name="..\\Scene_Analyzer\\sample_texts_normalized.csv"):
    """
    Load dreams from a CSV file.

    Args:
        file_name (str, optional): The path to the CSV file. 
        Defaults to "..\\Scene_Analyzer\\sample_texts_normalized.csv".

    Returns:
        pandas.DataFrame: A dataframe containing the loaded dreams.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_name = os.path.join(dir_path,"sample_texts_normalized.csv")
    dream_list = pd.read_csv(file_name, header=None)
    return dream_list


def get_samples(file_name="manual_scene_separation_data.txt"):
    """
    Open the file with the manually separated scenes and return a list of the separated dreams.

    file_name: Name of the file containing the separated scenes. Default is 'manual_scene_separation_data.txt'.
    Returns: 
        List of strings containing the separated dreams.
    """
    samples = []
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_name = os.path.join(dir_path,file_name)
    try:
        with open(file_name, "r") as f:
            data = f.read()
            samples = data.split("###")[1:-1]
            samples = [s.replace("IN:", "").strip() for s in samples]
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
    return samples


def build_prompt(
    dream, command="Give short visual descriptions of the scenes in the following:", n=3
):
    """
    Build the prompt for the API call.
    Start by giving n examples and their separation, then pass the command and the dream to be separated.
    
    Args:
        dream (str): The dream to be separated.
        command (str, optional): The command to be passed to the model.
        n (int, optional) = number of examples of manual separation to pass to the model
    """
    examples = ""
    samples = get_samples()
    
    # build the examples string from the manual separation data
    for i in range(0, min(len(samples), n)):
        examples += samples[i]
        examples += os.linesep

    # If we are passing examples in the prompt, we need to add "Examples:" to the prompt, otherwise we don't.
    if examples != "":
        prompt = f"Examples of dreams and scene seperation:\
    {examples.strip()}\
    {os.linesep}\
    {command}\
    {os.linesep}\
    {dream}"
    else:
        prompt = f"{command}{os.linesep}{dream}"
    return prompt


def load_latest_output():
    """
    Load the latest output from "out.txt" file.

    Reads the text from "out.txt" file and splits it into a list of strings based on the "OUT:" keyword. 
    It then further splits the list based on the "Scene" keyword, removes the line numbers and returns the list.
    
    Returns:
    List of strings containing the latest output from the "out.txt" file.
    """
    with open("out.txt", "r") as f:
        text = f.read()
        split_text = text.split("OUT:") # split the text based on the "OUT:" keyword
        gen_list = split_text[-1].split("Scene")[1:] # remove the first element
        gen_list = [re.sub(r"[0-9]\: ", "", x) for x in gen_list] # remove the line numbers
        gen_list[-1] = gen_list[-1].split('\n')[0]+'\n' # remove the last line
    return gen_list

def call_openai(
    dream,
    command="Give short visual descriptions of the scenes in the following:",
    test=False,
):
    """
    A function to call the OpenAI API and return a list of scenes resulting from the separation.

    dream = the dream to be analyzed
    command = the command to be passed to the model
    test = if True, the function will return a temporary text instead of calling the API
    """

    # temporary text to not spend tokens on the API
    if test == True:
        return load_latest_output()
    # model_engine = "text-curie-001"
    # model_engine = "davinci-002"
    model_engine = "text-davinci-003" # the best one so far for this task

    # API call to OpenAI GPT-3 using this schema:
    # https://beta.openai.com/docs/api-reference/completions/create
#     generated_text = "\n\n1. The first scene is of a person on an escalator, with plastic squares and water rolling along the side. The person later learns that they are filters. \n\n2. The second scene is of a large church where a mardi gras parade is taking place inside while mass is being celebrated in peace. \n\n3. The third scene is of a clerk coming to collect a bill which has already been paid. He has with him graded papers from a school, but the person does not see their son's name."
#     generated_text ='''1. Two dogs running across a sandy desert, with a person running away from them in the distance.
# 2. A child standing in a lush, green forest, looking around curiously.
# 3. An elderly woman sitting at a table in a brightly lit shopping mall,enjoying a cone of ice cream.'''
#     generated_text = """
#     Scene 1: This is scene 1.
#     Scene 2: This is scene 2.
#     3. This is line 3.
#     """
    prompt = build_prompt(dream.lstrip(), command, n=3)
    completions = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=256,
        n=1,
        stop=None,  # optional token that stops the generation
        temperature=0.45,  # not too high
    )

    # # # Print the generated text
    generated_text = completions.choices[0].text
    
    # Append the generated text to the output file to keep track of the results.
    with open("out.txt", "a+") as f:
        f.write(f"Prompt: {prompt}")
        ic(f"Prompt: {prompt}")
        f.write(f"Output: {generated_text}")
        ic(f"Output: {generated_text}")
        f.write(os.linesep)
        f.write(f"########################")
        ic(f"########################")
        f.write(os.linesep)
        
    def split_generated(generated_text):
        """
        Split the generated text into multiple scenes based on the occurrence of the word "Scene".
        Args:
            generated_text: The text to be split.
        Returns:
            A list of split scenes.
        """
        split_text = generated_text.split("Scene")[1:] # remove the first element because it's empty
        if len(split_text) != 0: 
            return split_text
        pattern = r"^\d+\."
        # Split the text using the pattern
        split_text = re.split(pattern, generated_text, flags=re.MULTILINE) 
        if len(split_text) == 0:
            split_text = generated_text.split('\n')
        return split_text[1:]
     
    
    gen_list = split_generated(generated_text)
    gen_list = [re.sub(r"[0-9](\:|\.) ", "", x) for x in gen_list]
    ic(gen_list)
    return gen_list


def separate():
    """
    return a random dream from the csv
    """
    # load the dreams from the csv
    dream_list = load_dreams(file_name="..\\Scene_Analyzer\\dream-export.csv")
    # show a random dream
    rand.seed(os.urandom(32))
    return dream_list[0][rand.randint(0, len(dream_list) - 1)]


def separate_random(test = False,command="Give short visual descriptions of the scenes in the following:"):
    """
    load a random dream from the csv and return the call to openai scene separator on it.
    """
    text = separate()
    ic(text)
    return call_openai(text, test=test,command=command)


if __name__ == "__main__":
    # Load a random dream from the csv and call the openai scene separator on it.
    separate_random(test=TEST)