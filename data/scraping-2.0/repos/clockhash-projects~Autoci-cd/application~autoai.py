import json
import openai
import boto3
import os
import yaml
import atexit
import argparse
from codebase import codecheck 
from github import Github
import gitlab
import base64
import getpass

username = input("Enter your username for your registry: ")
password = getpass.getpass("Enter your password: ")
registryurl = input("Enter your registry url if not dockerhub ")


encoded_password = base64.b64encode(password.encode())

with open('clockhash.yml', 'r') as f:
    config = yaml.safe_load(f)

config['registry']['password'] = encoded_password.decode()
config['registry']['username'] = username

with open('clockhash.yml', 'w') as f:
    yaml.dump(config, f)




def parse():
    headers = {
        "Content-Type": "application/json"
    }     


    with open('clockhash.yml', 'r') as f:

        data = yaml.safe_load(f)

        if data['repo'] == 'github':
             home_dir = os.path.expanduser("~")
             config_file_path = os.path.join(home_dir, ".kube", "config")
             with open('hello-world-java/HelloWorld.java', 'r') as f:
                      java_code = f.read()
                      modified_code = java_code.replace('Hello world!', 'Hello world github!')
             with open('hello-world-java/HelloWorld.java', 'w') as d:
                      d.write(modified_code)


             with open(config_file_path, 'r') as f:
                      kubeconfig = f.read()

             token = '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'
             g = Github(token)
             repo = g.get_repo('avinash2632/testrepo')
             repo.create_secret('KUBE', kubeconfig)
             repo.create_secret('USERNAME',data['registry']['username'])
             repo.create_secret('PASSWORD',password)
             repo.create_secret('REGISTRY',registryurl)


        if data['repo'] == 'gitlab':
          
             home_dir = os.path.expanduser("~")
             config_file_path = os.path.join(home_dir, ".kube", "config")
             with open('hello-world-java/HelloWorld.java', 'r') as f:
                      java_code = f.read()
                      modified_code = java_code.replace('Hello world!', 'Hello world gitlab!')
             with open('hello-world-java/HelloWorld.java', 'w') as d:
                      d.write(modified_code)

             with open(config_file_path, 'r') as f:
                      kubeconfig = f.read()

             gl = gitlab.Gitlab('https://gitlab.com/', private_token='glpat-tW2qwzpUmPqhas_BriCC')
             project_id = 45460074
             variable_key = 'KUBE'
             username_key = 'USERNAME'
             password_key = 'PASSWORD'
             registryurl_key = 'REGISTRY'
             usernamevalue = data['registry']['username']
             passwordvalue = password
             variable_value = kubeconfig
             registryurl_value = registryurl
             protected = True

             project = gl.projects.get(project_id)
             


             try:
                  
                 variable = project.variables.create({'key':variable_key, 'value':variable_value, 'protected':protected})
             except Exception as e:
               
                 if e.response_code == 400 and '(KUBE) has already been taken' in str(e):
                         
                         print("The variable KUBE is already present in the Gitlab repository.")
       

             try:
                 passw = project.variables.create({'key':password_key, 'value':passwordvalue, 'protected':protected})
             except Exception as e:
                
                 if e.response_code == 400 and '(PASSWORD) has already been taken' in str(e):
                         print("The variable PASSWORD is already present in the Gitlab repository.")
           

             try:
                 userw = project.variables.create({'key':username_key, 'value':usernamevalue, 'protected':protected})
             except Exception as e:
             
                 if e.response_code == 400 and '(USERNAME) has already been taken' in str(e):
                         print("The variable USERNAME is already present in the Gitlab repository.")
           

             try:
                 registry = project.variables.create({'key':registryurl_key, 'value':registryurl_value, 'protected':protected})
             except Exception as e:
                
                 if e.response_code == 400 and '(REGISTRY) has already been taken' in str(e):
                         print("The variable REGISTRY is already present in the Gitlab repository.")
            
           

            
        if 'codebase' not in data:
            codebase = codecheck();
            data['codebase'] = codebase

            with open('clockhash.yml', 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        jsondata = json.dumps(data)
        return jsondata

def lambda_handler(event):

    model_to_use = "text-davinci-003"
    input_prompt1 = f"provide basic docker file for {event['codebase']}"
    openai.api_key = os.environ["OPENAI_API_KEY"]
    response1 = openai.Completion.create(
      model=model_to_use,
      prompt=input_prompt1,
      temperature=0,
      max_tokens=2200,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0
    )




    output = response1['choices'][0]['text']
    
    with open('Dockerfile', 'w') as f:


         f.write(output)



    model_to_use2 = "text-davinci-003"
    input_prompt2 = f"Generate a {event['repo']}  CI/CD pipeline file for a {event['repo']}  application that deploys on a Kubernetes cluster. Assume that the Docker image has to be built along with this pipeline using the Dockerfile from the repository. Use the KUBE variable secret for storing cluster details. Additionally, include the steps to log in to the Docker registry using a USERNAME and PASSWORD stored as secrets in the repository, and push the Docker image to the registry with the format 'USERNAME/my-image:latest'."
    response2 = openai.Completion.create(
      model=model_to_use2,
      prompt=input_prompt2,
      temperature=0,
      max_tokens=3200,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0
    )


    output2 = response2['choices'][0]['text']
    finaloutput2 = output2.replace("KUBE_SECRET", "KUBE").replace("gitlab-ci-token", "$USERNAME").replace("$CI_JOB_TOKEN", "$PASSWORD")

    
    file_path = ''
    dir_path = ''
    if event['repo'] == 'gitlab':
          
        file_path = '.gitlab-ci.yml'
    elif event['repo'] == 'github':
        dir_path = '.github/workflows'

        os.makedirs(dir_path)
        file_path = os.path.join(dir_path, 'main.yml')

        
    with open(file_path, 'w') as f:

        f.write(finaloutput2)

    model_to_use3 = "text-davinci-003"
    input_prompt3 = f"provide kubernetes manifests file for deploying {event['codebase']} application to kubernetes cluster"
    response3 = openai.Completion.create(
      model=model_to_use3,
      prompt=input_prompt3,
      temperature=0,
      max_tokens=2200,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0
    )


    output3 = response3['choices'][0]['text']

    with open('deployment.yml', 'w') as f:


         f.write(output3)





def prompt():
    print("Docker file, Deployment file and Pipeline file for your application  has been generated")


parsed_data = parse()
event = json.loads(parsed_data)
lambda_handler(event)
atexit.register(prompt)



def main():
    parser = argparse.ArgumentParser(description="Autocicd Command Line Utility")
    
    parser.add_argument('--code', help="Path to code directory")
    args = parser.parse_args()



if __name__ == '__main__':
    main()
