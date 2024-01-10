import json
import os
import re
from langchain.chat_models import ChatAnthropic
from langchain.prompts import PromptTemplate
from datasets import load_dataset

example_indices = [27]

prompt = PromptTemplate.from_template("""
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
summarize the conversation to generate a clinical note with four sections: HISTORY OF PRESENT ILLNESS, PHYSICAL EXAM, RESULTS, ASSESSMENT AND PLAN.
The conversation is:
{conversation}

###Response to evaluate:
{summary}

###Reference Answer (Score 5):
{reference}

###Score Rubric:
[Is the model able to accurately and effectively summarize a medical conversation into a clinical note with four sections: HISTORY OF PRESENT ILLNESS, PHYSICAL EXAM, RESULTS, ASSESSMENT AND PLAN?]

Score 1: The summary utterly fails to reflect the conversation. It is incoherent, irrelevant, excessively verbose, or filled with hallucinations. There is a blatant disregard for standard clinical terminology, and critical information is omitted.

Score 2: The summary sporadically reflects elements of the conversation, but it frequently includes irrelevant or incoherent content. There is a noticeable lack of standard clinical terminology, verbosity is apparent, hallucinations are present, and critical information is often omitted.

Score 3: The summary generally captures the conversation accurately but occasionally includes irrelevant or incoherent content. It mostly uses standard clinical terms but can be verbose at times. Minor hallucinations may occur, and there might be instances of critical information being overlooked.

Score 4: The summary often accurately reflects the conversation, maintaining coherence and relevance throughout. There are minor cases of verbosity or use of non-standard clinical terms. The summary may have slight omissions or infrequent minor hallucinations.

Score 5: The summary flawlessly encapsulates the conversation, demonstrating complete coherence, relevance, and succinctness. It consistently employs standard clinical terminology, contains no hallucinations, and does not omit any critical information.

###Feedback:
""")


DATASET_PATH = '/mnt/c/Users/amangokrani/OneDrive - Microsoft/Personal/evaluations/outputs/train.gpt-4-1106-preview.pred.fullnote.json'
TEMP_PATH = '/mnt/c/Users/amangokrani/OneDrive - Microsoft/Personal/evaluations/outputs/temp/prompt-1'
dataset = load_dataset('json', data_files=DATASET_PATH)
dataset = dataset['train']


def parse_feedback(input_string):
    """
    Parses the string to extract the content that comes after '###Feedback:'.

    Args:
    input_string (str): The input string containing the '###Feedback:' marker.

    Returns:
    str: The substring that comes after '###Feedback:'.
    """
    # Define the marker
    marker = "Feedback:"

    # Find the index of the marker
    index = input_string.find(marker)

    # Check if the marker is found
    if index != -1:
        # Extract and return the substring that comes after the marker
        return input_string[index + len(marker):].strip()
    else:
        # Return an empty string if the marker is not found
        return ""


for index in example_indices: 
    example = dataset[index - 1]
    print("Processing file: ", example["file"])

    conversation = example["src"].replace('[doctor]', 'doctor:').replace('[patient]', 'patient:')
    conversation = re.sub(r'\s+([,.?])', r'\1', conversation)
    summary = example["pred"]
    reference = example["tgt"]
    
    #prompt = prompt.substitute(conversation=conversation, summary=summary, reference=reference)
    
    model = ChatAnthropic()
    chat = prompt | model
    output = chat.invoke({"conversation": conversation, "summary": summary, "reference": reference})
    output = parse_feedback(output.content)

    temp_file = os.path.join(TEMP_PATH, f"{example['file']}.json")
    import pdb; pdb.set_trace()
    with open(temp_file, "r") as file:
      data = json.load(file)
    if isinstance(data, list):
      # Append the new item to the list
      data.append(output)
    with open(temp_file, "w") as file:
      json.dump(data, file, indent=4)