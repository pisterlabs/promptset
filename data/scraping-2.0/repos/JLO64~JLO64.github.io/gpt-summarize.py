from bs4 import BeautifulSoup
from openai import OpenAI
import os, re, json

client = OpenAI()

def import_markdown(file_path):
    with open(file_path, 'r') as file:
        markdown_text = file.read()
    # print(file_path + " has been imported")
    return markdown_text

def remove_html(markdown):
    soup = BeautifulSoup(markdown, 'html.parser')

    tags_to_remove = ['figure', 'picture', 'p', 'script']
    for tag_type in tags_to_remove:
        tags_of_type = soup.find_all(tag_type)
        for tag in tags_of_type:
            tag.decompose()    
    
    return str(soup)

def convert_md_to_gpt(markdown_text):
    no_tags_markdown_text = remove_html(markdown_text).replace("\n\n\n", "")
    no_yaml_markdown_text = re.sub(r'---.*?---\s*', '', no_tags_markdown_text, flags=re.DOTALL)
    return no_yaml_markdown_text

def extract_description_line(text):
    lines = text.split('\n')
    for line in lines:
        if 'description: ' in line:
            return line.replace('description: ', "")
    return None


def generate_jsonl_training_file():
    writeup_data = []
    jsonl_export = []
    post_folder = "_posts/"

    for post_filename in os.listdir(post_folder):
        file_text = import_markdown(post_folder + post_filename)
        if ".md" in post_filename and "description: " in file_text:
            writeup_data.append( [convert_md_to_gpt(file_text), extract_description_line(file_text)] )
            
    for post_data in writeup_data:
        entry = {
            "messages": [
                {"role": "system","content": "You are a program designed to summarize blog posts in a certain tone and style of speech within two sentences."},
                {"role": "user","content": post_data[0]},
                {"role": "assistant","content": post_data[1]}
            ]
        }
        jsonl_export.append(entry)
    
    with open('description-data.jsonl', 'w') as f:
        for item in jsonl_export:
            f.write(json.dumps(item) + "\n")


# generate_jsonl_training_file()
            
def check_posts_for_no_desc():
    post_folder = "_posts/"

    for post_filename in os.listdir(post_folder):
        post_filepath = post_folder + post_filename
        file_text = import_markdown(post_filepath)
        if ".md" in post_filename and not "description: " in file_text:
            # gpt_generated_description = get_fine_tune_responce(convert_md_to_gpt(file_text))
            gpt_generated_description = select_gpt_generated_description(post_filename, convert_md_to_gpt(file_text))
            updated_post_text = add_description_to_post(file_text, gpt_generated_description)
            update_post_file(post_filepath, updated_post_text)

def select_gpt_generated_description(post_filename, converted_post_text):
    print("Please select the description for " + post_filename)
    num_of_times_to_generate = 3
    draft_gpt_generated_descriptions = []
    for i in range(num_of_times_to_generate):
        draft_gpt_generated_descriptions.append(get_fine_tune_responce(converted_post_text))
        print(str(i) + ": " + draft_gpt_generated_descriptions[i])
    
    flag = True
    input_value = None
    while flag:
        input_value = input("Please input a number: ")
        match_val = re.match("[-+]?\\d+$", input_value)
        if match_val is None:
            print("Please enter a valid integer.")
        elif int(input_value) >= num_of_times_to_generate:
            print("Please enter a valid option.")
        else:
            flag = False
    return draft_gpt_generated_descriptions[int(input_value)]

def add_description_to_post(text, description_to_add):
    old = '---'
    new = 'description: ' + description_to_add + '\n---'
    offset = text.index(old) + 1
    a = text[:offset] + text[offset:].replace(old, new, 1)
    return a

def update_post_file(filepath, file_data):
    with open(filepath, 'w') as f:
        f.write(file_data)


def get_fine_tune_responce(post_text):
    response = client.chat.completions.create(
      model="ft:gpt-3.5-turbo-1106:personal::8YpqxJ05",
      messages=[
        {"role": "system","content": "You are a program designed to summarize blog posts in a certain tone and style of speech within two sentences."},
        {"role": "user","content": post_text}
      ]
    )
    return(response.choices[0].message.content)

check_posts_for_no_desc()