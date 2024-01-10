# this script was used to apply the 'sentence' approach on the treebank test data. i used this for the fine-tuned models as well as gpt-3.5-tubro and gpt-4
import openai
import time

# replace with actual api key
openai.api_key = "API-KEY"

def make_splits(file):
    with open(file,"r") as f:
        content = f.readlines()

    splits = []
    split = []
    for row in content:
        # because not all datasets contained punctuation, i said a sentence to be 65 tokens long and gave this portion to the model in the prompt
        if len(split)<65:
            split.append(row.strip())
        else:
            split.append(row.strip())
            splits.append(" ".join(split))
            split = []
    return splits

cleaned_list = make_splits("../random-training-data-other-taggers/test_tok_udante.txt")
print(len(cleaned_list))

def perform_api_request_with_retry(sentence, model, max_retries=8):
    retry = 0
    while retry < max_retries:
        try:
            completion = openai.ChatCompletion.create(
                model = model,
                messages=[
                    {"role":"system","content": "You are a Latin linguist and part-of-speech tagging expert. You are using UPOS (universal part of speech tags). UPOS tags are ADJ,ADP,ADV,AUX,CCONJ,DET,INTJ,NOUN,NUM,PART,PRON,PROPN,PUNCT,SCONJ,SYM,VERB and X. X stands for 'other'."},
                    {"role":"user",
                    "content":f"Split the sentence only at every whitespace. Do not split the idividual tokens further. Return the UPOS tag for the tokens of the sentence in between the two dollar signs ${sentence}$ Return for every token in the sentence the token and the tag separated by a whitespace with no additional text (no explanations, no translations). After every pair add a newline character. The output should have the same number of elements as the input sentence, else redo the task."}
                    ]
                )
            print(completion)
            return completion.choices[0].message.content
        except Exception as e:
            # print(f"Error: {e}")
            retry += 1
            # Implement exponential backoff with a sleep duration that increases with each retry
            sleep_duration = 2 ** retry
            # print(f"Retry {retry}, sleeping for {sleep_duration} seconds...")
            time.sleep(sleep_duration)
    return None  # If all retries fail


# in the part below, change output file name and model name accordinly to model used

# Open your output file
with open("output-train10000-testset_udante.txt", "w") as out:
    for i,element in enumerate(cleaned_list):
        if element == "" or element == "\n" or element == " ":
            continue
        else:
            result = perform_api_request_with_retry(element, model="ft:gpt-3.5-turbo-0613:cl-uzh:train-10000:8HrmXwL4")
            if result:
                out.write(result)
                out.write("\n")

# # Open your output file
# with open("output-4-tokenized.txt", "w") as out:
#     for row in cleaned_list:
#         if row == "" or row == "\n" or row == " ":
#             continue
#         else:
#             sentence, tokens = row.split("|")
#             result = perform_api_request_with_retry(sentence, tokens, model="gpt-4")
#             if result:
#                 out.write(result)
#                 out.write("\n")





