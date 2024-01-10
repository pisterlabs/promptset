import openai
import json
import numpy as np
import random
import uuid

def generate_feedback(mission, raw_response, feedback_date, rating, rating_limits=False, rating_auto=False, model="gpt-4-1106-preview", temperature=1.2):
    
    if rating_auto and rating_limits:
        rating = np.random.randint(rating_limits[0], rating_limits[1]+1)
    else:
        rating=rating

    messages = [
        {
        "role" : "system",
        "content" : f"You simulate a project owner that just submitted a mission on the platform,\
        you benefit from an ai assistant that helps you to extract technical details from that mission so you can frame your needs rapidly and accurately.\
        This time, the AI was not performant enough. You give it a rating of {rating} stars out of 5 and you decide to fill a survey to give a feedback. Here is your original posted mission : {mission} "
        },

        {
            "role" : "user",
            "content" : json.loads(raw_response['choices'][0]['message']['function_call']['arguments'])['detail']
        }
        ]

    function = {"name" : "feedback",
    "description" : "A function that takes in a mission processed by the AI and returns a list of feedback given by the user",
    "parameters" : {
        "type" : "object",
        "properties" : {
            "user_rating" : {
                "type" : "integer",
                "description" : "The rating given by the user",

            },
            "user_comments" : {
                "type" : "string",
                "description" : "The comments given by the user",
                "minLength" : 30,
                "maxLength" : 500
            },
            "modification_details" : {
                "type" : "string",
                "description" : "The mission details given by the AI modified or rectified by the user it can be fully modified, partially or not at all if the user was satisfied",
            },


        },
        "required" : ["user_rating", "user_comments", "modification_details", "prompt_version"]

    }}
    response = openai.chat.completions.create(
        model = model,
        messages = messages,
        temperature = temperature,
        functions = [function],
        function_call = {"name" : "feedback"}
    )

    feedback_dict = json.loads(response.model_dump()['choices'][0]['message']['function_call']['arguments'])
    feedback_dict['id'] = uuid.uuid1()
    feedback_dict['created'] = feedback_date
    feedback_dict['prompt_version'] = np.random.choice(['v1', 'v2'])
    feedback_dict['mission_id'] = raw_response['id']
    
    return feedback_dict

def generate_mission(n=10, model="gpt-4-1106-preview", temperature=0.85):

    skills = [
        "Python", "JavaScript", "Java", "C++", "C#", "Ruby", "Go", "Rust", "Kotlin", "Swift", "TypeScript",
        "React", "Angular", "Vue.js", "Ember.js",
        "Django", "Flask", "Node.js", "Express.js", "Spring Boot", "Ruby on Rails",
        "SQL", "PostgreSQL", "MongoDB", "Redis", "Cassandra",
        "AWS", "Azure", "GCP",
        "Docker", "Kubernetes", "Jenkins", "Terraform",
        "Linux", "Unix", "Windows Server",
        "TensorFlow", "PyTorch", "Scikit-learn",
        "Cryptography", "Penetration Testing", "Firewall Management",
        "Adobe XD", "Sketch", "Figma",
        "GraphQL", "API REST", "WebRTC", "Blockchain", "Elasticsearch"
        ]
    

    tech_saviness = np.random.choice(['have a vague idea of tech', 'know a bit of tech', 'know a lot of tech', 'know nothing about tech'])
    rand_comb = random.sample(skills, random.randint(0, 3))
    
    messages = [{"role" : "system",
                "content" : f"""
                
                You are a mission offerer on a web dev/tech freelance platform. You {tech_saviness}. You want your mission to be done with either\
                combinations of these skills : {str(rand_comb)} or you might have no technological preference. Be concise and coherent.\
                Go straight to the point. \
                Don't write it in the form of a job offer but as you would like it to be done. Don't 
                        """}]
    

    response_generate = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=250,
    )

    text_response = response_generate.model_dump()["choices"][0]["message"]["content"]

    return text_response
