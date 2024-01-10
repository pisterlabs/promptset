#!/usr/bin/python3 

print("content-type: text/html")
print()

import cgi 
import subprocess

# import OPENAPI 
import json
import openai

# import gpt
from gpt import GPT
from gpt import Example

# OPENAPI KEY
openai.api_key = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

# Set the GPT Engine
gpt = GPT(engine="davinci",
          temperature=0.5,
          max_tokens=100)

# Add Examples to be trained
gpt.add_example(Example('Launch a myweb deployment with httpd image.', 
                        'kubectl create deployment myweb --image=httpd'))

gpt.add_example(Example('Run a test deployment with vimal13/apache-webserver-php as image', 
                        'kubectl create deployment test --image=vimal13/apache-webserver-php'))

gpt.add_example(Example('Run a webapptest deployment with vimal13/apache-webserver-php as image', 
                        'kubectl create deployment webapptest --image=vimal13/apache-webserver-php'))

gpt.add_example(Example('Run a webapptesting deployment with httpd as image', 
                        'kubectl create deployment webapptesting --image=httpd'))

gpt.add_example(Example('Launch a deployment with name as webapp and image as httpd', 
                        'kubectl create deployment webapp --image=httpd'))

gpt.add_example(Example('Create a pod with name as testing and image as httpd', 
                        'kubectl run testing --image=httpd'))

gpt.add_example(Example('Launch a pod with webpod as name and vimal13/apache-webserver-php as image', 
                        'kubectl run webpod --image=vimal13/apache-webserver-php'))

gpt.add_example(Example('Launch a pod with webtest as name and httpd as image', 
                        'kubectl run webtest --image=httpd'))

gpt.add_example(Example('Delete deployment with name test', 
                        'kubectl delete deployment test'))

gpt.add_example(Example('Delete deployment with name webapp', 
                        'kubectl delete deployment webapp'))

gpt.add_example(Example('Delete a pod with name webtest', 
                        'kubectl delete pod webtest'))

gpt.add_example(Example('Expose the deployment test as NodePort type and on port 80', 
                        'kubectl expose deployment test --port=80 --type=NodePort'))

gpt.add_example(Example('Expose the deployment webtest as External LoadBalancer type and on port 80', 
                        'kubectl expose deployment webtest --port=80 --type=LoadBalancer'))

gpt.add_example(Example('Expose the deployment webapp as ClusterIP type and on port 80', 
                        'kubectl expose deployment webapp --port=80 --type=ClusterIP'))

gpt.add_example(Example('Create 5 replicas of test deployment', 
                        'kubectl scale deployment test --replicas=5'))

gpt.add_example(Example('Create 3 replicas of webapp deployment', 
                        'kubectl scale deployment webapp --replicas=3'))

gpt.add_example(Example('Delete all resources of Kubernetes', 
                        'kubectl delete all --all'))

gpt.add_example(Example('Get the list of deployments', 
                        'kubectl get deployments'))

gpt.add_example(Example('Get the list of services', 
                        'kubectl get svc'))

gpt.add_example(Example('List all the pods', 
                        'kubectl get pods'))


f = cgi.FieldStorage()
prompt = f.getvalue('x')

# Getting the Prediction 
output = gpt.submit_request(prompt)
res = output.choices[0].text

cmd = res.split("output")[1].split(":")[1].strip()
cmd = cmd + " --kubeconfig /root/kubews/admin.conf"
print(cmd)
print()
output = subprocess.getoutput('sudo ' + cmd)
print(output)
