import openai
import json
import random
import re

from typing import List, Dict

MAX_LEN = 4096

openai.api_key = ""
random_names = ['Abigail Cooper', 'Samuel Davis', 'Olivia Rodriguez', 'Michael Campbell', 'Emily Watson',
         'David Parker', 'Sarah Foster', 'Christopher Ramirez', 'Madison Reed', 'William James', 
         'Jessica Hayes', 'Matthew Hernandez', 'Emma Mitchell', 'Anthony Barnes', 'Isabella Wright', 
         'James Lee', 'Sophia Phillips', 'Daniel Coleman', 'Elizabeth Flores', 'Benjamin Black', 'Natalie Ortiz', 
         'Ryan Nguyen', 'Grace Collins', 'Tyler Green', 'Lily Bennett', 'Jonathan King', 'Ava Patel', 'Nicholas Taylor', 
         'Charlotte Cox', 'Samuel Sullivan', 'Mia Baker', 'Andrew Brown', 'Abigail Ross', 'Joseph Cooper', 'Harper Evans', 
         'Timothy Hill', 'Evelyn Perry', 'Brandon Tucker', 'Victoria West', 'Jacob Rivera', 'Amelia Peterson', 'Ethan Reyes', 
         'Madison Price', 'Alexander Carter', 'Sofia Diaz', 'Brian Wright', 'Scarlett Bailey', 'Zachary Kim', 'Chloe Fisher', 
         'Kevin Stewart', 'Addison Crawford', 'Ryan Powell', 'Aubrey Hunter', 'Thomas Wilson', 'Hannah Gomez', 
         'Gabriel Sullivan', 'Zoe Ortiz', 'Elijah Brooks', 'Ellie Collins', 'Nicholas Parker', 'Ava Price', 
         'Benjamin Green', 'Mia Murphy', 'Sean Nelson', 'Avery Hayes', 'Justin Wood', 'Harper Torres', 'Caleb Johnson', 
         'Leah Baker', 'Richard Kim', 'Gabriella Peterson', 'Charles Stewart', 'Aaliyah Ramirez', 'Eric Ward', 
         'Layla Adams', 'Nathan Bailey', 'Hailey Bell', 'Adam Watson', 'Kaylee Brooks', 'Kyle Collins', 
         'Arianna Wright', 'Robert Cooper', 'Peyton Torres', 'Aaron Kim', 'Audrey Carter', 'Jason Hernandez', 
         'Mackenzie Sullivan', 'Matthew Bailey', 'Taylor Coleman', 'Steven Torres', 'Alexa Perez', 'Isaac Davis', 
         'Lily Sanders', 'Brian Ramirez', 'Brianna Parker', 'Frank Flores', 'Madison Carter', 'Keith King', 
         'Isabelle Campbell', 'Vincent Gonzalez']

def anonymize(resume: Dict[str, str]) -> Dict[str, str]:
    """
    :param resume: a dictionary representing a resume.
    :return: a dictionary representing a resume with names in each and every field replaced by random names.
    """
    anonymized_resume = {}
    label_fields = ['T1_Label', 'T2_Label']
    titles = ['Mr.', 'Ms.', 'Miss', 'Mrs.', 'Dr.']
    suffixes = ['Jr.', 'Sr.']

    for field in resume:
        if field in label_fields:
            anonymized_resume[field] = resume[field]
            continue

        content = "Give me all person in the following text using the following list format: [NAME, NAME, NAME] and output NNF if no person are found"
        content += "\n\n" + resume[field]
        if len(content) >= MAX_LEN:
            content = content[:MAX_LEN]
        response = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": content},
            ]
        )

        parsed_content = []
        for choice in response.choices:
            content = choice["message"]["content"]
            for name in parse_responses(content):
                parsed_content.append(name)

        substitutions = []
        for name in parsed_content:
            random_index = random.randint(0, 99)
            random_name = random_names[random_index].split(" ")
            splitted_name = name.split(" ")
            actual_name_len = len(splitted_name)
            for title in titles:
                if splitted_name[0] == title:
                    random_name.insert(0, title)
                    actual_name_len -= 1
            for suffix in suffixes:
                if splitted_name[-1] == suffix:
                    random_name.append(suffix)
                    actual_name_len -= 1
            if actual_name_len == 1:
                del random_name[1]
            random_name = " ".join(random_name)
            substitutions.append(random_name)
            
        
        text = resume[field]
        for i in range(0, len(parsed_content)):
            text = text.replace(parsed_content[i], substitutions[i])
        anonymized_resume[field] = text
        
    return anonymized_resume

def parse_responses(response: str) -> List[str]:
    response.replace("\n", "")
    if response == "NNF":
        return []
    bidx = 0
    eidx = 0
    for idx, character in enumerate(response):
        if character == '[':
            bidx = idx
        elif character == ']':
            eidx = idx
    parsed_response = response[bidx+1:eidx].split(",")
    invalid_tags = ["NNF", "", "N/A"]
    for tag in invalid_tags:
        while tag in parsed_response:
            parsed_response.remove(tag)
    return parsed_response