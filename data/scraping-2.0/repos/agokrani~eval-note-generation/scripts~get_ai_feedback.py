import json
import os
import re
import random
from collections import Counter
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.prompts import PromptTemplate
from datasets import load_dataset



def max_vote_with_random_tie(input_list):
    # Count the votes
    vote_count = Counter(input_list)

    # Find the maximum vote count
    max_votes = max(vote_count.values())

    # Create a list of items with the maximum vote count
    candidates = [item for item, count in vote_count.items() if count == max_votes]

    # Randomly select one if there is a tie
    return random.choice(candidates) if len(candidates) > 1 else candidates[0]

def get_preference_details_from_file(file_path):
    preferred_file = open(file_path, "r")
    pref_data = preferred_file.read()
    pref_data = pref_data.split('\n')
    pref_data = dict_data = {item.split(': ')[0]: int(item.split(': ')[1]) for item in pref_data[:]}
    num_files_to_ignore = len(pref_data)
    return pref_data, num_files_to_ignore

PROMPT = PromptTemplate.from_template(
    """
    Given the conversation between doctor and patient, the model generated clinical note, along with the clinical note provided by the doctor, and 2 evaluation feedbacks of the model
    generated clinical note, you are asked to evaluate the evaluation feedbacks and select the one that you think is more appropriate.
    
    Here is the conversation:
    {conversation}

    Here is the model generated clinical note:
    {summary}

    Here is the clinical note provided by the doctor:
    {reference}

    Here are the evaluation feedbacks:
    {feedbacks}

    Please select the feedback that you think is more appropriate. Only write the index of the feedback. For example, if you think the first feedback is more appropriate, 
    please write 1. If you think the second feedback is more appropriate, please write 2.
    
    ==========================================================
    NOTE: Please do not write any other text in the feedback.
    ==========================================================
    """
)

ANTHROPIC_PROMPT = PromptTemplate.from_template("""
Given the conversation between doctor and patient, the model generated clinical note, along with the clinical note provided by the doctor, and 2 evaluation feedbacks of the model generated clinical note, you are asked to evaluate the evaluation feedbacks and select the one that you think is more appropriate.  Please select the feedback that you think is more appropriate. Only write the index of the feedback. For example, if you think the first feedback is more appropriate, please write 1. If you think the second feedback is more appropriate, please write 2.
You ALWAYS follow these guidelines when writing your response:
<guidelines>
- Your output will always be just the answer which is just the index of feedback. 1 or 2
- Do not write any explanation 
</guidelines>

Human: 
<conversation>
{conversation}
</conversation>
    
<model-gen-note>
{summary}
</model-gen-note>

<doctor-note>
{reference}
</doctor-note>

<feedbacks>
{feedbacks}
</feedbacks>

Which feedback is preferred? 


Assistant: [integer here]""")

DATASET_PATH = '/mnt/c/Users/amangokrani/OneDrive - Microsoft/Personal/evaluations/outputs/train.gpt-4-1106-preview.pred.fullnote.json'
dataset = load_dataset('json', data_files=DATASET_PATH)
dataset = dataset['train']

TEMP_PATH = '/mnt/c/Users/amangokrani/OneDrive - Microsoft/Personal/evaluations/outputs/temp/prompt-1'
TEMP_PREFERRED_FEEDBACKS_PATH = '/mnt/c/Users/amangokrani/OneDrive - Microsoft/Personal/evaluations/outputs/temp/preferred.txt'

pref_feedbacks = []
rejected_feedbacks = []

pref_data, num_files_to_ignore = get_preference_details_from_file(TEMP_PREFERRED_FEEDBACKS_PATH)
preferred_file = open(TEMP_PREFERRED_FEEDBACKS_PATH, "a")

for i, example in enumerate(dataset):
    temp_file = os.path.join(TEMP_PATH, f"{example['file']}.json") 
    with open(temp_file, "r") as file:
        data = json.load(file)

    if i <= num_files_to_ignore:
        print("Skipping file: ", example["file"])
        pref_feedbacks.append(data[pref_data[example["file"]]])    
        rejected_feedbacks.append(data[not pref_data[example["file"]]])
        continue

    print("Processing file: ", example["file"])
    
    conversation = example["src"].replace('[doctor]', 'doctor:').replace('[patient]', 'patient:')
    conversation = re.sub(r'\s+([,.?])', r'\1', conversation)
    summary = example["pred"]
    reference = example["tgt"]

    feedbacks = ""
    feedbacks_reversed = ""
    for index in range(len(data)):
        feedbacks += f"Feedback {index+1}: \n{data[index]}\n\n"
        feedbacks_reversed += f"Feedback {index+1}: \n{data[-1-index]}\n\n"
    
    oai_model = ChatOpenAI(model='gpt-4-1106-preview')
    aai_model = ChatAnthropic(model='claude-2.1', temperature=0.0)
    
    chat_oai = PROMPT | oai_model
    chat_aai = ANTHROPIC_PROMPT | aai_model
    preferred_list = []

    out = chat_oai.invoke({"conversation": conversation, "summary": summary, "reference": reference, "feedbacks": feedbacks}) # 1 or 2 then index - 1 = 0 or 1 
    preferred_list.append(int(out.content) - 1)

    out = chat_oai.invoke({"conversation": conversation, "summary": summary, "reference": reference, "feedbacks": feedbacks_reversed}) # len(data)-index
    preferred_list.append(len(data) - int(out.content))
    
    out = chat_aai.invoke({"conversation": conversation, "summary": summary, "reference": reference, "feedbacks": feedbacks})
    try: 
        preferred_list.append(int(out.content) - 1)
    except:
        preferred_list.append(int(out.content.split('\n\n')[0]) -1)

    out = chat_aai.invoke({"conversation": conversation, "summary": summary, "reference": reference, "feedbacks": feedbacks_reversed})
    try: 
        preferred_list.append(len(data) - int(out.content))
    except:
        preferred_list.append(len(data) - int(out.content.split('\n\n')[0]))
    pref_out = max_vote_with_random_tie(preferred_list)
    
    preferred_file.write(f"{example['file']}: {pref_out}\n")
    
    rejected_out = int(not pref_out)

    pref_feedbacks.append(data[pref_out])    
    rejected_feedbacks.append(data[rejected_out])


dataset = dataset.add_column("chosen", pref_feedbacks)
dataset = dataset.add_column("rejected", rejected_feedbacks)

dataset.to_json("/mnt/c/Users/amangokrani/OneDrive - Microsoft/Personal/evaluations/outputs/train.dpo.gpt-4.claude-2.1.max-rand.json")
