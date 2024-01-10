import subprocess
import os
import shutil
import pandas as pd
import openai
from docx import Document

username = input("First and last name: ")
firstname = username.split()[0].lower()

if not os.path.isfile(f"{firstname}-data/user_data.csv"): # If the user hasn't filled out the questionnaire
    # Call Shane's questionnaire program
    os.chdir("./questionnaire")
    subprocess.call(["g++", "main.cpp", "questionnaire.cpp", "-o", "questionnaire"])
    subprocess.call(["./questionnaire"])

    # Turn csv file into txt file
    questionnaire = pd.read_csv('user_data.csv')
    questionnaire = questionnaire.drop(questionnaire.columns[0], axis=1)
    with open("questionnaire.txt", 'w') as f:
        for i in range(questionnaire.shape[0]):
            for j in range(questionnaire.shape[1]):
                f.write(str(questionnaire.iat[i, j]) + '\n')

    # Save any user data to it's own folder
    os.chdir("../")
    if not os.path.exists(f"{firstname}-data"):
        os.mkdir(f"{firstname}-data")
    source_dir = './questionnaire'
    target_dir = f'./{firstname}-data'
    filenames = ['user_data.csv', 'questionnaire.txt']
    for filename in filenames:
        source = os.path.join(source_dir, filename)
        target = os.path.join(target_dir, filename)
        shutil.move(source, target)

bst = False

if bst:
    # Call Celia's BST skill word extraction program
    os.chdir("./BST")
    subprocess.call(["g++", "BST.cpp", "-o", "BST"])
    subprocess.call(["./BST"])
    source_dir = './BST'

else:
    # Call Shane's heap skill word extraction program
    os.chdir("./heap")
    subprocess.call(["g++", "heap.cpp", "-o", "heap"])
    subprocess.call(["./heap"])
    source_dir = './heap'

# Save any user data to it's own folder
os.chdir("../")
if not os.path.exists(f"{firstname}-data"):
    os.mkdir(f"{firstname}-data")
target_dir = f'./{firstname}-data'
filenames = ['job-post.txt', 'skill-list.txt']
for filename in filenames:
    source = os.path.join(source_dir, filename)
    target = os.path.join(target_dir, filename)
    shutil.move(source, target)

# Load in the data from files
os.chdir(f"./{firstname}-data")
with open("questionnaire.txt") as file:
    user_data = file.read()

with open("job-post.txt") as file:
    job_post = file.read()

with open("skill-list.txt") as file:
    skill_list = file.read()

# Setup chatGPT
print("Enter your Open AI API key:")
openai.api_key = input()

# # Use chatGPT to find more skill words
# my_message = job_post + "\n Above is a job posting, please extract any relevant skills that you think the employer might be looking for. In particular look for repeated words. Each skill word should be a single item in a comma separated list. Do not provide anything beyond this list of skills."
# response = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[{"role": "user","content": my_message}]
#     )
# reply = response["choices"][0]["message"]["content"]
# final_skill_str = skill_list + reply
# rec_skills = [word.strip() for word in final_skill_str.split(',')]

# # Go over the list and see which skills the user actually has
# rec_skills = [word.strip() for word in skill_list.split(',')]
# user_skills = []
# print("Our program has gone through and picked out skills from the job posting that we think the employer might be looking for. Let's go over them. Remember to be generous with yourself as many skill words can be vague, but don't be afraid to say you don't have a skill if you don't have it.")
# for skill in rec_skills:
#     print(f"Would you say that you have the skill \"{skill}\"? (y/n)")
#     line = input()
#     if line.strip().lower() == 'y':
#         user_skills.append(skill)
# user_skills_str = ", ".join(user_skills)

# Make the resume
print("We're working on your resume. Please wait...")
my_message = "Given the following resume information...\n" + "Name\n" + username + "\n" + user_data + "\nMake a resume using as many of the following skill words as possible through out the resume...\n" + skill_list.replace(",", "")
response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user","content": my_message}]
            )
reply = response["choices"][0]["message"]["content"]

# Print and save the resume
print("Here is a draft of your resume populated with words that will help it reach a real human:\n")
print(reply)
doc = Document()
for paragraph in reply.split("\n"):
    if paragraph:
        doc.add_paragraph(paragraph)
doc.save("resume-draft.docx")

# Future development
# - We need to add sources to the readme.md in the datasets folder for Celia's datasets even if it's just us saying we web scraped them
# - Automatic formating
#   - Automatic formatting will require the questionare to be totaly overhalled
#   - That way we can build the resume from the formating up. That's the only way to do it, but I didn't know till finishing the project
#   - Rebuilding the questionare will give us a chance to streamline it to as few questions as possible
#   - We can also add Better UX (i.e. not just command line interface)