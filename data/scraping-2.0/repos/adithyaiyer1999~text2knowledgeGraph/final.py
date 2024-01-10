from openai import OpenAI
import openai
import json2tree
import subprocess

#prompt = "The user will provide a textual passage. Your task is to analyze the passage and respond with a JSON formatted structure that breaks down the passage into hierarchical ideas, similar to headings and subheadings in a document. For each identified section or idea, create a nested structure in JSON. Start with broader themes or main points as top-level elements, and break them down into finer details or sub-points. Ensure the JSON output clearly represents the hierarchy and organization of ideas in the passage, from the most general to the most specific." ## give your prompt here
prompt = "You are a assigned a task to build a knowledge graph. Based on the text provided you have to create a JSON output such that key will represent all the significant elements of the text and values would represent the summary of key. Break down the values into more granular level information creating a tree or graph based hierarchy. Create a JSON representation for the same."
api_key="open-ai-api-key" ## give your api_key here
prompt_for_graph_update="You will be given two inputs following this command, first is the a json string and second is a paragraph to update in the json string. The json tree is a knowledge tree which puts the information in a form of heirarchial structure, making the information text into a granular level json representation. Your task is to take in the existing json text and append the new paragraph given into the form of json representation into the existing json. You cannot lose information of the old json. Following are the json and paragraph."


'''This function takes text as input and returns corresponding json string'''
def give_json_string(paragraph):
    return query_gpt_turbo(prompt,paragraph)

'''This function takes new_information_to_update and old_json_string as input and returns corresponding updated json string'''
def update_existing_graph(new_information_to_update,old_json_string):
    return query_gpt_turbo(prompt_for_graph_update, new_information_to_update,old_json_string)

'''This function takes json_string as input and returns corresponding html string'''
def create_html_from_json(json_input):

    # Write the json to the file because the library only takes file as input
    with open("example.json", 'w') as file:
        file.write(json_input)

    # Command and arguments
    command = "json2tree"
    json_input = "-j example.json"
    html_output = "-o output.html"
    tree_type = "-t 1"

    # Full command
    full_command = f"{command} {json_input} {html_output} {tree_type}"

    # Run the command
    subprocess.run(full_command, shell=True, check=True)

    with open('output.html','r') as file:
        output_html_text = file.read()

    return output_html_text

def query_gpt_turbo(prompt,content,old_json_string="",model="gpt-3.5-turbo",api_key=api_key):
    openai.api_key = api_key
    client = OpenAI(
        api_key=api_key,
    )
    chat_completion=create_chat_completion(client,old_json_string,prompt,content,model)
    final_output = chat_completion.choices[0].message.content
    return final_output
def create_chat_completion(client,old_json_string,prompt,content,model):
    ## This is for graph update
    if old_json_string != "":
        # print("inside new one")
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system",
                 "content": prompt},
                {
                    "role": "user",
                    "content": old_json_string,
                },
                {
                    "role": "user",
                    "content": content,
                }
            ],
            model=model,
        )
    ## This is for normal json output
    else:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system",
                 "content": prompt},
                {
                    "role": "user",
                    "content": content,
                }
            ],
            model=model,
        )
    return chat_completion

if __name__ == '__main__':
    with open('input_text', 'r') as file:
        file_contents = file.read()
    with open('old_json','r') as file:
        old_json = file.read()
    with open('new_information_to_update','r') as file:
        new_information_to_update = file.read()
    old_json=give_json_string(file_contents)
    # print("old query done")
    # # print("old json:",old_json)
    # print(update_existing_graph(new_information_to_update,old_json))
    # print("updation done")
    # print(create_html_from_json(old_json))




