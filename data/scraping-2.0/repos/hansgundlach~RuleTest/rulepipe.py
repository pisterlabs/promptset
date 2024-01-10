# %%
import openai
from utils import *
import random
from dotenv import load_dotenv
import os

# %%
# load environment key
load_dotenv()

api_key = os.getenv("API_KEY")


# %%
# generates anagram of a given yes and no word
def gen_anagram_example(num_examples=10, yesword="bbbbbccc", noword="cccccbbb"):
    # examine performance as number of in-context samples is increased.
    example_string = ""
    counter = 0
    while counter < num_examples:
        example_string += rand_an(yesword) + ":Yes" + "\n"
        counter += 1
        example_string += rand_an(noword) + ": No" + "\n"
        counter += 1
    # print(example_string)
    return example_string


# %%


# generate text with flag code words
def gen_flag_example(num_examples=10, input_file="tinyshak.txt"):
    example_string = ""
    counter = 0
    while counter < num_examples:
        example_string += (
            f"Statement {counter}: "
            + random_string_from_file(input_file, length=100)
            + "(XYP)"
            + "\n Classification: No"
            + "\n \n \n "
        )
        counter += 1
        example_string += (
            f"Statement {counter}: "
            + random_string_from_file(input_file, length=100)
            + "(XYZ)"
            + " \n Classification: Yes"
            + "\n \n \n"
        )
        counter += 1
    return example_string


# %%
start_prompt = """"
Here are statements with their labels added: 
"""


add_label_prompt = """ \n Now add labels to the statements I will give you using the same pattern as the examples above. Please do not add anything other then the classification label as your response. The class label is one word. Please do not return anything other than the class label. You must return only one word. You will respond with a single word that is either Yes or No:"""

# %%
# measure performance for programatic tasks
# extract_answer from each answer
file_path = "FrozenExamples/bctask2.txt"
with open(file_path, "r") as f:
    file_contents = f.read()

labeled_sentences = file_contents
yesword = "bbbbbbbbbbc"
noword = "ccccccccccb"
# labeled_setntences = gen_anagram_example(10)
# labeled_sentences = gen_anagram_example(5, yesword=yesword, noword=noword)

prompt = start_prompt + labeled_sentences + add_label_prompt
print(prompt)

# %%

# test performance on anagram task
num_examples = 10
results = []
for i in range(num_examples):
    output = ask_openai(prompt + rand_an(yesword))
    results.append(output)
print(results)
# %%

# now examine if model can articulat bc-task.
file_path = "FrozenExamples/bctask2.txt"
with open(file_path, "r") as f:
    file_contents = f.read()

labeled_sentences = file_contents

prompt_art = """"
Here is a set of statements with their labels. \n """
question = """ \n can you articulate the rule behind the labels?"""
print(prompt_art + labeled_sentences + question)
print(ask_openai(prompt_art + labeled_sentences + question))

# ====================================================================================================
# Flag Detection Task


# %%

file_path = "FrozenExamples/shakshak.txt"
with open(file_path, "r") as f:
    file_contents = f.read()

# (XYZ) corresponds to yes
# (XYP) corresponds to no
labeled_sentences = file_contents
# labeled_sentences = gen_flag_example(5)
prompt = start_prompt + labeled_sentences + add_label_prompt
print(prompt)

# examine performanc on code-word injection task
num_examples = 20
results = []
for i in range(num_examples):
    output = ask_openai(
        prompt + random_string_from_file("tinyshak.txt", length=100) + "(XYP)"
    )
    results.append(output)
print(results)

# %%

# test acticulation of flag task
file_path = "FrozenExamples/shakshak.txt"
with open(file_path, "r") as f:
    file_contents = f.read()

# labeled_sentences = file_contents
labeled_sentences = gen_flag_example(15)

prompt_art = """"
Here is a set of statements with their labels."""
question = """can you articulate the rule behind the labels?"""
print(ask_openai(prompt_art + labeled_sentences + question))


#
# start of investigation into COT prompting
prompt_art = """"
Here is a set of statements with their labels. \n """
# question = """Can you articulate the simple pattern behind the labels?
# """
# question= """ Can you tell me the rule for each label?"""

file_path = "FrozenExamples/shakshak.txt"
with open(file_path, "r") as f:
    file_contents = f.read()

labeled_sentences = file_contents
# labeled_sentences = gen_anagram_example(10)
labeled_sentences = gen_flag_example(10)
question = """Think step by step about how to find the rule behind the labels?"""
answer = ask_openai(prompt_art + labeled_sentences + question)
COTprompt = (
    prompt_art
    + labeled_sentences
    + question
    + f" \n Assistant Answer: {answer} "
    + "Given the information above, what is your final answer for the general rule behind the labels?"
)

# print(COTprompt)
final_answer = ask_openai(COTprompt)
# %%
full_text = COTprompt + f" \n Final Answer: {final_answer} "
print(full_text)

# %%
