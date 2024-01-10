import os

import openai
import json
import re
import PyPDF2


def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text


def read_prompt_from_file(prompt_file):
    with open(prompt_file, "r") as file:
        return file.read().strip()

def read_api_key(api_key_file):
    with open(api_key_file, "r") as file:
        return file.read().strip()

def resume_parser_openai(input_text, prompt_file):

    prompt = read_prompt_from_file(prompt_file)

    api_key_file = "api_key.txt"
    api_key = read_api_key(api_key_file)
    openai.api_key = api_key

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": input_text}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        # model="gpt-3.5-turbo-16k",
        messages=messages,
        max_tokens=1000,
        n=1,
        stop=None
    )

    output = response['choices'][0]['message']['content']

    # Save the response in the 'response.txt' file
    with open("response.txt", "w") as file:
        file.write(output)

    return output


def resume_parser_json(input):

    # Extract Experiences
    experiences = re.findall(r'Name: (.+)\n- Position: (.+)\n- Duration: (.+)\n- Description:((?:.+\n)+)', input)
    experiences_list = []
    for experience in experiences:
        company, position, duration, description = experience
        description = re.findall(r'(.+)', description)
        experiences_list.append({
            "company": company,
            "position": position,
            "duration": duration,
            "description": description
        })

    # Extract Project Work
    project_work = re.findall(r'Title: (.+)\n- Project Description: (.+)\n- Technologies: (.+)', input)
    project_work_list = []
    for project in project_work:
        project_heading, project_desc, technologies = project
        project_work_list.append({
            "project_heading": project_heading,
            "project_desc": project_desc,
            "technologies": technologies,
            "gitUrl": "",
            "extUrl": "",
            "imgSrc": ""
        })

    # Extract Contact Info
    try:
        contact_info = re.findall(r'Contact Info:\n- Email: (.+)', input)[0]
    except:
        contact_info = "",""
        
    # Extract Intro Info
    try:
        intro_info = re.findall(r'Intro Info\n- Name: (.+)\n- Status: (.+)', input)
        name, status = intro_info[0]
    except:
        name, status = "",""


    try:
        first_para = re.search(r'First Paragraph: (.+)\n', input).group(1)
    except AttributeError:
        first_para = ""

    try:
        second_para = re.search(r'Second Paragraph: (.+)\n', input).group(1)
    except AttributeError:
        second_para = ""

    # Extract LinkedIn
    try:
        linkedin = re.findall(r'LinkedIn: (.+)', input)[0]
    except:
        linkedin = ""

    # Extract Github
    try:
        github = re.findall(r'Github: (.+)', input)[0]
    except:
        github = ""

    # Extract Email ID
    try:
        email_id = re.findall(r'Email: (.+)', input)[0]
    except:
        email_id = ""

    # Extract Summary
    try:
        summary = re.search(r'Summary:\s+([\s\S]+?)\n\n', input).group(1)
    except:
        summary = ""
        
    # Extract Skills
    try:
        skills_text = re.search(r'Skills:\s+([\s\S]+)', input).group(1)
        skills = re.findall(r'- (.+)', skills_text)
    except:
        skills = ""

    # Extract headerData
    header_data = {
        "resumeSrc": "",
        "logoSrc": "/assets/favicon-512x512.png"
    }

    # Extract footerData
    footer_data = {
        "gitUrl": "https://github.com/r1shabhpahwa/ResumeParserOpenAI",
        "creditContent": "UWindsor - School of Computer Science",
        "creditUrl": "https://uwindsor.ca/",
        "selfCreditContent": "Built by <strong>Group 28</strong>",
        "gitRepo": "r1shabhpahwa/ResumeParserOpenAI"
    }


    # Extract socialMediaLinks
    social_media_links = {
        "githubUrl": github,
        "leetcodeUrl": "",
        "instaUrl": "",
        "twitterUrl": "",
        "linkedinUrl": linkedin
    }

    # Extract emails
    emails = [email_id]

    thanks_note = """Thank you for visiting my portfolio! <br /><br />
    Please don't hesitate to contact me through email or other platforms. I look forward to hearing from you soon!"""

    # Create the JSON object
    data = {
        "userData": {
            "experiences": experiences_list,
            "projectWork": project_work_list,
            "contactInfo": {"email": contact_info, "content":thanks_note},
            "introInfo": {"name": name, "status": status, "displayPic":"","summary": summary},
            "aboutInfo": {"firstPara": first_para, "secondPara": second_para, "displayPic":"", "skillsList":skills}
        },
        "headerData": header_data,
        "footerData": footer_data,
        "socialMediaLinks": social_media_links,
        "emails": emails
    }

    # Convert to JSON
    json_output = json.dumps(data, indent=4)

    return json_output


def read_file_content(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' could not be found.")
        return ""
    except Exception as e:
        print(f"Error: An error occurred while reading the file '{file_path}': {e}")
        return ""


if __name__ == '__main__':

    # Extract Text from PDF
    input_text = extract_text_from_pdf('resume_2.pdf')

    # Extract Data from Resume Text using OpenAI and append to response.txt
    resume_parser_openai(input_text, 'prompt.txt')

    # Read response.txt
    response = read_file_content('response.txt')

    # Parse text using Regex and populate JSON
    json_output = resume_parser_json(response)

    print(json_output)
    with open("data.json", "w") as outfile:
        outfile.write(json_output)
    
