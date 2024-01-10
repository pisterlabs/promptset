import json
import re
from openai import OpenAI
import tiktoken
import time
# pip install openai
# pip install tiktoken

token_count = 0
count = 0
encoding = tiktoken.get_encoding("cl100k_base") # needed for counting tokens to optimize tokens per minute 
def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Set your OpenAI API key
client = OpenAI(api_key='API-KEY-HERE')

# Get string of prompt from file
prompt_file_path = 'prompt.txt'
try:
    with open(prompt_file_path, 'r') as file:
        prompt_str = file.read()
except FileNotFoundError:
    print(f"The file '{prompt_file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# Define regular expressions for extracting data between square brackets without double quotes
question_label_pattern = re.compile(r'Question Label:\s*\["(.*?)"\]')
context_label_pattern = re.compile(r'Context Labels:\s*\["(.*?)"\]')
keyword_label_pattern = re.compile(r'Keywords:\s*\["(.*?)"\]')
# Define a regular expression pattern to extract the reponse 'content'
content_pattern = re.compile(r"content='(.*?)'", re.DOTALL)
# Function to remove double quotes from extracted data
def remove_quotes(match):
    if match:
        return match.group(1).replace('", "', ', ')

def generate_labels(prompt):
    # Make API request
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,    
        max_tokens=4095,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # Handle response
    if 'error' in response:
        print(f"Error: {response['error']['message']}")
        return None
    # Extract generated text
    # Use the pattern to find the content
    match = content_pattern.search(str(response))
    if match:
        generated_text = match.group(1)
    else:
        generated_text = "error, retry question later"
    return generated_text


def process_json(json_data):
    global token_count
    global count
    for item in json_data:
        if "Conversations" in item:
            for conversation in item["Conversations"]:
                con = False
                content = conversation.get("Prompt")
                prompt = prompt_str + content + '"'

                if "QuestionLabels" not in conversation or "ContextLabels" not in conversation or "KeywordLabels" not in conversation:
                    while con == False:
                        try:
                            labels = generate_labels(prompt)
                            con = True
                            # print("success")
                            # print(content[0:30])
                            time.sleep(0.5)
                        except Exception as e:
                            # print("fail")
                            # print(e)
                            print(content)
                            time.sleep(0.5)

                    # print(labels)
                    # print("-----------------------------------------------")

                    # Extract data using regular expressions
                    question_label_match = question_label_pattern.search(labels)
                    context_label_match = context_label_pattern.search(labels)
                    keyword_label_match = keyword_label_pattern.search(labels)
                    if question_label_match and "QuestionLabels" not in conversation:
                        conversation["QuestionLabels"] = remove_quotes(question_label_match)
                    if context_label_match and "ContextLabels" not in conversation:
                        conversation["ContextLabels"] = remove_quotes(context_label_match)
                    if keyword_label_match and "KeywordLabels" not in conversation:
                        conversation["KeywordLabels"] = remove_quotes(keyword_label_match)
                    if not question_label_match or not context_label_match or not keyword_label_match:
                        conversation["ErrorGPT"] = "something went wrong"
                    else:
                        conversation.pop("ErrorGPT", None) 

                else:
                    conversation.pop("ErrorGPT", None)

        else:
            con = False
            content = item.get("Body")
            prompt = prompt_str + content + '"'

            if "QuestionLabels" not in item or "ContextLabels" not in item or "KeywordLabels" not in item:
                while con == False:
                    try:
                        labels = generate_labels(prompt)
                        con = True
                        # print("success")
                        # print(content[0:30])
                        time.sleep(0.5)
                    except Exception as e:
                        # print("fail")
                        # print(e)
                        # print(content)
                        time.sleep(0.5)
                
                # print(labels)
                # print("-----------------------------------------------")

                # Extract data using regular expressions
                question_label_match = question_label_pattern.search(labels)
                context_label_match = context_label_pattern.search(labels)
                keyword_label_match = keyword_label_pattern.search(labels)
                if question_label_match and "QuestionLabels" not in item:
                    item["QuestionLabels"] = remove_quotes(question_label_match)
                if context_label_match and "ContextLabels" not in item:
                    item["ContextLabels"] = remove_quotes(context_label_match)
                if keyword_label_match and "KeywordLabels" not in item:
                    item["KeywordLabels"] = remove_quotes(keyword_label_match)
                if not question_label_match or not context_label_match or not keyword_label_match:
                    item["ErrorGPT"] = "something went wrong"
                else:
                    item.pop("ErrorGPT", None)

            else:
                item.pop("ErrorGPT", None)

                


# ------------------------------------------------------------------------
# Read and process the first JSON file (ChatGPT)
try:
    with open("refined_devgpt.json", "r") as file:
        devgpt_data = json.load(file)
        process_json(devgpt_data)

    # Read and process the second JSON file (StackOverflow)
    with open("refined_so.json", "r") as file:
        so_data = json.load(file)
        process_json(so_data)

except KeyboardInterrupt:    
    print("Exited but still saved label progress.")
except Exception as e:
    print("Exited but still saved label progress.")

# Combine the processed data from both JSON files
combined_data = devgpt_data + so_data
# ------------------------------------------------------------------------
# # uncomment this section to make sure all labels are applied and/or the program cancelled after first run, comment out section above
# try:
#     with open("combined_data.json", "r") as file:
#         data = json.load(file)
#         process_json(data)
# except KeyboardInterrupt:    
#     print("Exited but still saved label progress.")
# except Exception as e:
#     print("Exited but still saved label progress.")
# combined_data = data
# ------------------------------------------------------------------------



# Write the combined data to a new JSON file
with open("combined_data.json", "w") as file:
    json.dump(combined_data, file, indent=2)

print("Combined data to combined_data.json.")
