from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables from .env file
load_dotenv(".env")

# Get your API key from the environment variables
api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is available
if api_key:
    # Initialize OpenAI client with the API key
    client = OpenAI(api_key=api_key)
else:
    print("API key not found in environment variables.")


def getGPT(email, subject):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": """Pretend you are a consultants at analyzing email.You are given a email about software, determine whether this email is about new job posting or application, if the following is about new job posting use this JSON format:
 "JobList": "Yes",
{ Jobs:[   
     "Company":
     "Role":
     "Location":
    "URL":
]} 
else if it about application update use the following JSON format, and DO NOT include the "JobList":
"Application": "Yes",
{
    "Company":
    "Status": "Applied"/"Rejected"/"Interview"/"Online Assignment"
    "Role": }
if its anything else return the response in following JSON format and nothing else:
{
    "Joblist":  "No"
    "Application": "No"
} If you are given email from the following assume that I already finish my application: 
"Thanks for your interest in (role) on WayUp! Have you completed the second part of the application process on (company)"



""",
            },
            {
                "role": "user",
                "content": "Subject: " + subject + "\nEmail: " + email,
            },
        ],
        temperature=0.5,
    )

    # Extract the response
    generated_text = response.choices[0].message.content

    return generated_text
