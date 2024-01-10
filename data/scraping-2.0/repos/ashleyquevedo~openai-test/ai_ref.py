# pip install scipy
# pip install tenacity
# pip install tiktoken
# pip install termcolor
# pip install openai
# pip install requests

import json
import openai
import requests
import os
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored

# temp
OPENAI_KEY = "INSERT KEY HERE"
GPT_MODEL = "gpt-3.5-turbo-0613"


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(
    messages, functions=None, function_call=None, model=GPT_MODEL, temperature=0
):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + OPENAI_KEY,
    }
    json_data = {"model": model, "messages": messages}
    if functions is not None:
        json_data.update({"functions": functions})
    if function_call is not None:
        json_data.update({"function_call": function_call})
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


def pretty_print_conversation(messages):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta",
    }
    formatted_messages = []
    for message in messages:
        if message["role"] == "system":
            formatted_messages.append(f"system: {message['content']}\n")
        elif message["role"] == "user":
            formatted_messages.append(f"user: {message['content']}\n")
        elif message["role"] == "assistant" and message.get("function_call"):
            formatted_messages.append(f"assistant: {message['function_call']}\n")
        elif message["role"] == "assistant" and not message.get("function_call"):
            formatted_messages.append(f"assistant: {message['content']}\n")
        elif message["role"] == "function":
            formatted_messages.append(
                f"function ({message['name']}): {message['content']}\n"
            )
    for formatted_message in formatted_messages:
        print(
            colored(
                formatted_message,
                role_to_color[
                    messages[formatted_messages.index(formatted_message)]["role"]
                ],
            )
        )


functions = [
    {
        "name": "extract_query_attributes",
        "description": "Given a person's resume, extract the person's work experience in months that is applicable to full stack development, frontend development, backend development, software development and software engineering. Extract the different programming languages and frameworks that the person knows and grade them by knowledge level low medium or high.",
        "parameters": {
            "type": "object",
            "properties": {
                "work_experience": {
                    "type": "array",
                    "description": "an array of work experiences that is can be related to full stack development, frontend development, backend development, software development, software engineering or other. The job_title is the actual title of his job, the skill_set is the most relatable work experience I just mentioned, and the time_worked is the amount of time the person was in that position in months",
                    "items": {
                        "type": "object",
                        "properties": {
                            "job_title": {
                                "type": "string",
                                "description": "actual title of the person job",
                            },
                            "skill_set": {
                                "type": "string",
                                "description": "the most relatable work experience in a enum of full stack development, frontend development, backend development, software development or software engineering",
                                "enum": [
                                    "full stack development",
                                    "frontend development",
                                    "backend development",
                                    "software engineering",
                                    "other",
                                ],
                            },
                            "months_worked": {
                                "type": "number",
                                "description": "the amount of time the person was in that position in a string format. Try to interpret the string and extract the number of months, rounded up the person has worked in that position",
                            },
                        },
                    },
                },
                "skills": {
                    "type": "array",
                    "description": "An array of the different programming languages and frameworks that the person knows and grade them by knowledge level low medium or high ",
                    "items": {
                        "type": "object",
                        "properties": {
                            "language": {
                                "type": "string",
                                "description": "a programming language or framework that the person knows",
                            },
                            "values": {
                                "type": "string",
                                "description": "the amount of knowledge that a person knows graded by low, medium, or high. Use work experience to determine this information as well.",
                                "enum": ["low", "medium", "high"],
                            },
                        },
                    },
                },
            },
        },
    }
]

# (skills, education, experience, contact info, projects, objective, misc)

messages = []
messages.append(
    {
        "role": "system",
        "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.",
    }
)
messages.append(
    {
        "role": "user",
        "content": "Manmohit Matharu\
416-400-2666 | manmohit.matharu@gmail.com | LinkedIn | GitHub\
Education\
Lighthouse Labs Toronto, ON\
Diploma in Full Stack Web Development June 2022\
Ryerson University Toronto, ON\
Bachelor of Commerce in Finance and International Business June 2012\
Technical Skills\
Working knowledge: JavaScript, React, Express, Node.js, PostgreSQL, Jira, MySQL, Git, HTML, CSS/SCSS,\
Storybook JS, REST, Object Oriented Programming\
Exposure to: Python, AWS Lambda, TypeScript, Ruby, Axios, Mocha, Chai, Twilio, Bootstrap, Jest, Cypress, Vercel,\
jQuery\
Experience\
Supply Chain Planner Oct 2018 – Oct 2020\
Martin Brower Mississauga, ON\
• Managed inventory for McDonald’s, Chick-fil-A, Chipotle, and Panera Bread distribution centers in the US\
• Implemented new processes and utilized Power BI to track metrics, leading to a 23% improvement in planner KPIs\
• Reduced manual errors by 7% and increased planner productivity by 13% after standardizing out-of-office coverage\
and generating a supplier item master list\
• Minimized supply disruptions 13% by implementing an inventory shortage tracker accessible to various stakeholders\
Senior Analyst Nov 2012 – Sept 2017\
Aon Toronto, ON\
• Mentored a team of 4-5 indirect reports ensuring seamless delivery of services to 10+ clients, including HP,\
Honeywell, Siemens, and Cintas, representing over 10,000 employees and $3.3+ billion CAD in insurance premiums\
• Increased cross-selling opportunities 15% by leading weekly client calls mitigating HR data risks and identifying\
trends which resulted in increased client satisfaction by 23%, and reduced escalations by 20%\
• Addressed incoming client concerns, worked with internal departments to resolve any system defects, and developed\
and documented 10+ SOPs, providing training to junior analysts, resulting in a 25% reduction in turnaround time\
for their work, client inquiries, and employee escalations\
Projects\
Delve | GitHub | JavaScript, React, SCSS, PostgreSQL, Twilio June 2022\
• Developed a group study app allowing users to create rooms, add members, set session goals, and track progress\
metrics as they watch synchronized video lectures\
• Implemented a live chat feature with status indicators using WebSocket and webcam integration with Twilio\
• Utilized PostgreSQL to handle account and room creation, and generate chat logs\
• Adopted a responsive design utilizing accessible color palettes for an inclusive user experience on all devices\
Honeywell Voice of Customer | Aon June 2016 – Sept 2016\
• Identified 15 gaps based on feedback from 500+ employees on their benefits enrollment post-acquisition\
• Presented a comprehensive fit-gap analysis to Honeywell, outlining recommendations for 9 system improvements\
• Obtained approval and collaborated with the development team implementing changes in QC/PROD environments\
• Resulted in a 25% reduction to inquiries and increased participation in enrollment across the Honeywell population\
Hewlett Packard Workforce Reduction (WFR) | Aon May 2016 – Aug 2016\
• Automated portions of the WFR process, reducing payment delinquency and saving the client $50,000 annually\
• Drafted change requirements working with various stakeholders in an effort to eliminate multiple points of failure\
• Streamlined the process by digitizing key components and revising legal documentation which resulted in a 25%\
reduction in inquiries to the call centre\
• Resulted in a 25% reduction in process times, and a 77% reduction to premature benefits coverage termination",
    }
)

chat_response = chat_completion_request(messages, functions=functions)
assistant_message = chat_response.json()
print(
    json.loads(assistant_message["choices"][0]["message"]["function_call"]["arguments"])
)
