import openai
import os
from dotenv import load_dotenv, find_dotenv
# Read local .env file for API key
_ = load_dotenv(find_dotenv())  

openai.api_key = os.environ['OPENAI_API_KEY']

def load_resume():
    # Load my resume from a text file
    with open('static/resources/resume.txt', 'r') as file:
        resume = file.read()
    return resume

def generate_proposal(user_entry):
    
    # This is the delimiter that will be used to separate the job description from the proposal
    delimiter = '***' 
    
    #Go get my resume
    resume = load_resume()

     # If the user asks a question, you will answer it.
    
    # Setting up the system to know what to expect 
    system_message = f"""You are an assistant who accepts a job proposal or questions from a user. You will use the resume provided 
    to add experience. The resume is: {resume}. 
    If the user provides a job proposal, you will accept it.
    You will use the job description provided
    to write a job proposal.
    The job proposal or question will be delimited with \
    {delimiter} characters."""
    
    messages =  [  
    {'role':'system',
     'content':system_message},    
    {'role':'user', 
     'content': f"{delimiter}{user_entry}{delimiter}"}, 
    ] 
    
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages,
        max_tokens=500,
        # n=1,
        # stop=None,
        temperature=0.7,
    )
    try: 
        answer = response.choices[0]['message']['content'].replace('\n', '<br>')
    except:
        answer = 'Sorry, I did not understand that. Please try again.'
    
    return answer
                      