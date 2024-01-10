import yaml
import openai
import os
import sys
from openai import OpenAI
# Load ther configuration file
with open(r'C:\Users\submi\OneDrive\Documents\pythonProjects\rpiq\RPIQ\src\config.yml', 'r') as file:
    config = yaml.safe_load(file)
client = OpenAI(api_key=config['openai_api_key'])
      
title = ""
# a genorator function to iterate through the files in a directory
def get_files(path=r"C:\Users\submi\OneDrive\Documents\pythonProjects\rpiq\RPIQ\src\rpiq\data\catagories\Soft Skills"):
    global title
    for root, dirs, files in os.walk(path):
        for file in files:
            title = file.split('.')[0]
            read_file = open(os.path.join(root, file), 'r').read()
            yield read_file
# define how to use the generator function
def use_get_files():
    content = get_files()
    for i in range(5):
        
        print(next(content))
        return next(content)
# use_get_files()
# # Usage example

# # Get the files

#load the yalm template
template = open(r'C:\Users\submi\OneDrive\Documents\pythonProjects\rpiq\RPIQ\src\rpiq\data\examples\YAML_Template_for_Roleplay_Scenarios.yaml', 'r').read()


# Define the function to upload the file
def upload_file(template=template):
    # Get the file path
    # Get the file name
    # Get the file extension
    content = use_get_files()
    print(len(content))
    if len(content) > 0:
            
        completion = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "you are a completionest with creative writing skills of a book ninja. please use you content creating expertise to help me write a yaml file for my project. to fill any elements thats not provided in the yaml file to be co nverted to the templates schema. please use the following template to fill in the blanks. " + template + ""},
                {"role": "user", "content":"please convert the following to adhere to the scgema above returning only the new yaml file " + use_get_files() + ""}
            ]
        )

        # Get the assistant's reply
        reply = completion.choices[0].message.content

        return reply


# Usage example
if __name__ == "__main__":
    # Get the assistant's reply
    for i in range(7):
        reply = upload_file()
        # Print the assistant's reply
        print(reply)
        # Save the assistant's reply to a file
        with open(r'C:\Users\submi\OneDrive\Documents\pythonProjects\rpiq\RPIQ\src\rpiq\data\catagories\revamped\\'+title+".yaml", 'w') as file:
            file.write(reply)
        #remove the file from the directory
        os.remove(r'C:\Users\submi\OneDrive\Documents\pythonProjects\rpiq\RPIQ\src\rpiq\data\catagories\Soft Skills\\'+title+'.yaml')        #then go to the     next filereply = upload_file()

