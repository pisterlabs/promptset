import os
import openai
from dotenv import load_dotenv
import regex
import time
import re

pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')
my_keys = ['name', 'email_id', 'phone_no', 'total_experience', 'current_employer', 'current_designation', 'skills',
           'current_skills', 'location']
dotenv_path = os.path.join(os.getcwd(), '.env')
load_dotenv(dotenv_path)
openai.api_key = os.environ.get("OPENAI_API_KEY")


def Data_Parser(data):
    try:
        prompt = f'''
            Please extract the following details from the provided resume text and present them in Python JSON format:
            - Name
            - Email ID
            - Phone Number
            - Location
            - Total Experience (expressed in years)
            - Current Employer (Identify the candidate's most recent workplace, focusing on keywords like "Current Employer," "Recent Job," or "Present Company")
            - Current Designation
            - Skills
            - Current Skills
            
            For each detail, please use the following keys in the JSON:
            - Name: name (string)
            - Email ID: email_id (string)
            - Phone Number: phone_no (string)
            - Location: location (string)
            - Total Experience: total_experience (integer in years)
            - Current Employer: current_employer (string)
            - Current Designation: current_designation (string)
            - Skills: skills (list of strings)
            - Current  Skills: current_skills (list of strings)
            
            To improve the accuracy of extracting the "name" field, follow these strategies:
                1. Search for sections or lines that may contain personal information, such as "Personal Details," "Contact Information," or "Profile."
                2. Extract the text from these sections and analyze patterns that typically represent names.
                3. Use regular expressions to match common name patterns, while considering the presence of special characters and diacritics.
                4. Normalize Unicode characters (e.g., NFKD) to handle diacritics and special characters effectively.
           To improve the accuracy of extracting the "Current Employer," follow these strategies:
                1. Locate sections or headings related to work experience, such as "Work History," "Employment," or "Professional Experience."
                2. Extract the text within these sections and capture the surrounding context, including job titles and date ranges.
                3. Search for specific keywords like "Current Employer," "Present Company," or "Recent Job" within these sections.
                4. Account for formatting variations using regular expressions to match potential variations of the keywords.
                5. Be cautious with keywords like "LinkedIn" which might be present in contexts other than current employment. Consider filtering out these false positives by checking adjacent job titles and date ranges.
                6. Utilize multiple context clues, such as adjacent job titles and date ranges, to accurately identify the current employer.
                
            To extract the candidate's "Current Skills":
                1. Focus on the work experience sections related to the identified current employer.
                2. Identify keywords or phrases that indicate skills, such as "Skills," "Technologies Used," or similar headings.
                3. Extract the text within these sections and parse for relevant skills.
    
            When identifying the current employer, focus on sections explicitly related to work experience. Exclude comments in the JSON output. Provide only JSON data.
            
            Resume:-
            
            {data[:6000]}



        
        '''

        def make_openai_request():
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=1000  # Adjust as needed
            )
            return response

        while True:
            try:
                parsed_data = make_openai_request()
                break  # If successful, break the loop
            except openai.error.OpenAIError as e:
                if "rate_limit_exceeded" in str(e):
                    wait_time = 20  # seconds
                    print(f"Rate limit exceeded. Waiting for {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise e

        parsed_data = parsed_data.choices[0].text
        return parsed_data
    except Exception as err:
        msg = Data_Parser(data)
        return msg


def clean_name(name):
    cleaned_name = re.sub(r'[^a-zA-Z\s\.\-]', '', name)
    return cleaned_name.strip()


def response_validation(response):
    data = {}
    try:
        data = eval(pattern.findall(response)[0])
        if data["name"]:
            data["name"] = clean_name(data["name"])

        filtered_data = {key: data[key] for key in my_keys}

        return filtered_data
    except Exception as err:
        return data
