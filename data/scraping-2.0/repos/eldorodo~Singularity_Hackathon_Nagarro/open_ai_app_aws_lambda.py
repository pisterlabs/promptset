import json
import os
print(os.getcwd())
import numpy as np
from numpy.linalg import norm
import boto3
import yarl
from botocore.exceptions import ClientError
import openai
import csv
import ast
import math
#sys.path.append("/mnt/access")

s3 = boto3.client('s3')


#### One time activity to save embeddings ####

mental_health_words = ['anxiety', 'depression', 'stress', 'panic', 'trauma', 'therapy', 
                       'medication', 'psychologist', 'psychiatrist', 'counseling', 'mental illness', 
                       'bipolar', 'suicide', 'obsessive-compulsive disorder', 'PTSD', 'phobia', 'addiction', 
                       'compulsive behavior', 'schizophrenia', 'mania', 'delusion', 'paranoia', 'psychosis', 
                       'personality disorder', 'borderline personality disorder', 'narcissism', 
                       'dissociative identity disorder', 'eating disorder', 'anorexia', 'bulimia', 
                       'binge eating disorder', 'body dysmorphic disorder', 'psychosomatic disorder', 
                      'conversion disorder', 'somatoform disorder', 'hypochondria', 'psychogenic amnesia',
                         'memory loss', 'dementia', 'Alzheimer’s disease', 'obsession', 'insomnia', 
                         'sleep disorder', 'nightmare', 'sleep apnea', 'restless legs syndrome', 
                         'antisocial personality disorder', 'narcissistic personality disorder', 
                         'borderline personality disorder', 'schizoid personality disorder', 
                         'schizotypal personality disorder', 'avoidant personality disorder', 
                         'dependent personality disorder', 'obsessive-compulsive personality disorder', 
                         'paranoid personality disorder', 'depersonalization disorder', 
                         'derealization disorder', 'manic episode', 'hypomanic episode', 'mixed episode',
                           'major depressive episode', 'hallucination', 'delusion', 'suicidal ideation',
                             'mania', 'psychotic episode', 'postpartum depression', 'perinatal depression', 
                             'premenstrual dysphoric disorder', 'gender dysphoria', 'gender identity disorder',
                              'identity crisis', 'body image', 'self-esteem', 'self-harm', 'self-injury', 
                               'suicide prevention', 'anger management', 'cognitive behavioral therapy', 
                               'dialectical behavior therapy', 'exposure therapy', 'interpersonal therapy',
                                 'psychodynamic therapy', 'group therapy', 'family therapy', 'support groups', 
                                 'mindfulness', 'meditation', 'breathing exercises', 'yoga', 'exercise',
                                   'nutrition', 'self-care', 'relaxation techniques']
                                   
                                   
def download_from_s3(bucket_name,key):
    
    response = s3.get_object(Bucket=bucket_name, Key=key)
    contents = response['Body'].read().decode('utf-8')
    
    return

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
   
  
def cosine_similarity(v1,mental_health_embeddings_dict):
    # Compute cosine similarity between the embeddings

    for word, v2 in mental_health_embeddings_dict.items():
        #v2 = mental_health_embeddings_dict[i]
        v1, v2 = np.array(v1), np.array(eval(v2))
        cosine_sim = np.dot(v1,v2)/(norm(v1)*norm(v2))
        if(cosine_sim > 0.8):
            return True, word
    return False, "none of the topics"
    

def lambda_handler(event, context):
    
    print("type of event",type(event))
    print("event print", event)

    if('body' in event): #From Postman
        event = json.loads(event['body'])
        user_id = event['user_id']
        prompt = event['prompt']
        openai_key = event['openai_key']
        
    elif('queryStringParameters' in event): #Get Query
        #event = json.loads(event['queryStringParameters'])
        user_id = event['queryStringParameters']['user_id']
        prompt = event['queryStringParameters']['prompt']
        openai_key = event['queryStringParameters']['openai_key']
        
    else:
        user_id = event['user_id']
        prompt = event['prompt']
        openai_key = event['openai_key']
        
        
    # TODO implement
    messages = [
    {"role": "system", "content": "You are a helpful and kind AI Assistant on mental health topics"},
    ]
    
    bucket_name = 'singularityhackathonearthmind'
    object_key = 'messages' + '_' + user_id + '.json'
    
    try:
        # Read the JSON file from the S3 bucket
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        json_str = response['Body'].read().decode('utf-8')

        # Convert the JSON string to a list of dictionaries
        messages = json.loads(json_str)

    except ClientError as e:
        # If the object doesn't exist, save the default messages list as a JSON file in S3
        #if e.response['Error']['Code'] == 'NoSuchKey':
        print(f"The object with key '{object_key}' doesn't exist in bucket '{bucket_name}'. Saving default messages list.")
        json_str = json.dumps(messages)
        s3.put_object(Body=json_str, Bucket=bucket_name, Key=object_key)

        # Convert the JSON string to a list of dictionaries
        messages = json.loads(json_str)
        #else:
            # If there's some other error, re-raise the exception
            #raise e
            
    #if 'prompt' not in event or 'openai_key' not in event:
    #        return {
    #    'statusCode': 400,
    #    'body': 'Missing input data'
    #}

    if(not isinstance(prompt, str)):
                    return {
        'statusCode': 400,
        'body': 'Prompt input must be a string'
    }

    openai.api_key = openai_key
    
    # Read the JSON file from the S3 bucket
    response = s3.get_object(Bucket=bucket_name, Key='mental_health_embeddings_dict.json')
    json_str = response['Body'].read().decode('utf-8')

    # Convert the JSON string to a list of dictionaries
    mental_health_embeddings_dict = json.loads(json_str)

    sim, topic = cosine_similarity(get_embedding(prompt), mental_health_embeddings_dict)  
    print(sim, topic)
    if(not sim):
        reply = "Sorry, I am AI bot to assist on the specific topic of mental health"
    else:
        messages.append({"role": "user", "content": prompt})
        chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages
            )

        reply = chat.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
    
    return {
        'statusCode': 200,
        'body': json.dumps(reply)
    }
