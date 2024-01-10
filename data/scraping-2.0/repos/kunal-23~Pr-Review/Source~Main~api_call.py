import requests
import sys
import json
import openai

def chat_with_chatgpt(prompt):
    conversation = []
    conversation.append({'role':'user','content':str({prompt})})
    review = openai.ChatCompletion.create(model=model_id, messages=conversation)
    return review.choices[0].message.content

def post_review(headers, url2, chatgpt_response, fileName):
    pull_response = requests.get(url2, headers=headers)
    pr_info = pull_response.json()

    # Post review as a comment on the Github PR
    comment_url = pr_info['comments_url']
    comment_body = "Automated review using OpenAI for file {}:\n {}\n".format(fileName,chatgpt_response)
    git_response = requests.post(comment_url, headers=headers, json={'body': comment_body})
    if git_response.status_code == 201:
        print('\nReview posted successfully')
    else:
        print('\nError posting review: {}'.format(git_response.text))

openai.api_key ="<gpt_key>"
model_id="gpt-3.5-turbo"

# Authenticate with Github API
auth_token = '<github_key>'
headers = {'Authorization':'Token ' + auth_token}

#PR details
owner = "Raghav-Bajaj"
repo = "Pr-Review"
base = 'main'
head = 'my-branch'
pull_number=4

#Endpoints
url1 = f'https://api.github.com/repos/{owner}/{repo}/compare/{base}...{head}'

url2 = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}"

#PR details
pr_response = requests.get(url1, headers=headers)

if pr_response.status_code == 200:
    diffs = pr_response.json()["files"]
    for i, diff in enumerate(diffs):
        chatgpt_response = ''
        add_changes = []
        delete_changes = []
        original_file = ''
        patch = diff["patch"]
        fileName = diff["filename"]
        if(diff["status"] == 'modified' or diff["status"] == 'deleted'):
            print(f'\nModified or deleted file :{fileName}');
            url4 = f"https://raw.githubusercontent.com/{owner}/{repo}/{base}/{fileName}"
            response = requests.get(url4, headers=headers)
            original_file= response.text
        else:
            print(f'\nNew file is being added :{fileName}')  
          
        for line in patch.split('\n'):
            if line.startswith('+'):
                add_changes.append(line[1:])
            elif line.startswith('-'):
                delete_changes.append(line[1:]) 

        added_changes = '\n'.join(add_changes)
        removed_changes = '\n'.join(delete_changes)

        comment_p = f"Evalute the changes on the basis of the provided information and please help me do a brief code review. The programmimg language used is JAVA. If any bug risk and improvement suggestions are welcome and tell if the code is meeting the coding standards and quality. 3 variables will be provided namely originalFile, added_changes and removed_changes. originalFile will contain the current contents of the file present in the branch for which PR is being raised. added_changes will contain the changes being added to the originalFile as part of PR and removed_changes will contain the changes removed from this file. If added_changes is empty means nothing is added or if removed_changes is empty means nothing is removed from the file. Similarly if originalFile is empty it means the entire file is new. Below provided information is related to only {fileName} file."
        
        prompt = f"{comment_p} \n originalFile: {original_file} \n added_changes: {added_changes} \n removed_changes: {removed_changes}"
        print(f"prompt: {prompt}")
        chatgpt_response = chat_with_chatgpt(prompt)
        print("Chatgpt:::>",end='\n')
        for i in chatgpt_response:
            sys.stdout.write(i)
        
        post_review(headers, url2, chatgpt_response, fileName)
        
else:
    print(f"Failed to get diff, status code: {pr_response.status_code}")
