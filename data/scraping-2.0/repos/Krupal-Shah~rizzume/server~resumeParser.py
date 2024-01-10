from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import os
from pdfminer.high_level import extract_text
import subprocess

API_KEY = ''
openai = OpenAI(api_key=API_KEY)

def textgeneration(prompt):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
       messages=[
            {"role": "system", "content": "You are a prompt fulfiller bot who who does what is asked by the prompt diligently. You will recieve very specific instructions on what to do by the prompt and you need to follow that. Treat the prompts as rules."},
            {"role": "user", "content": prompt}],
        )

    return response.choices[0].message.content

# getting text from job posting and classifying with GPT
def scrape_text_jobposting(url):
    try:
        # Send an HTTP request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract text content
            text_content = soup.get_text()
            # print("Text from soup: ",text_content)

            prompt = f'You are given this job description {text_content}, give me a json file with the following parameters only: requirements, skills, job description, responsibilities, programming languages .'

            content = textgeneration(prompt)
            # print("GPT simplified: ",content)
            #print(content)
            return content

        else:
            # print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
            return None

    except Exception as e:
        # print(f"An error occurred: {e}")
        return None


def extract_resume(resume_location):
    resume_dict = {"GenDet": [], "Education": [], "Experience": [], "Projects": [], "TechSkills": []}
    text = extract_text(resume_location)

    prompt_gen = f"Here is a resume: {text}.Extract the general details like name and links"
    content_gen = textgeneration(prompt_gen)
    resume_dict["GenDet"].append(content_gen)

    prompt_ed = f"Here is a resume: {text}.Extract the education part only"
    content_ed = textgeneration(prompt_ed)
    resume_dict["Education"].append(content_ed)

    prompt_exp = f"Here is a resume: {text}.Extract the Experience part only if exists"
    if prompt_exp:
        content_exp = textgeneration(prompt_exp)
        resume_dict["Experience"].append(content_exp)

    prompt_pro = f"Here is a resume: {text}.Extract the Projects part only"
    if prompt_pro:
        content_pro = textgeneration(prompt_pro)
        resume_dict["Projects"].append(content_pro)

    prompt_tech = f"Here is a resume: {text}.Extract the Technical skills part only"
    if prompt_tech:
        content_tech = textgeneration(prompt_tech)
        resume_dict["TechSkills"].append(content_tech)

    return resume_dict


def extract_jobdescription(jobposting):
    extrated_posting = {"requirements": [], "skills": [], "job description": [], "responsibilities": [],
                        "programming languages": []}
    # prompt = f"I have {extrated_posting} dictionary and {jobposting}.Parse the job description and return the given dictionary with the paramters filled in"
    # result = textgeneration(prompt)
    # data_dict = json.loads(result)
    # print(data_dict,type(data_dict))
    extrated_posting["requirements"] = textgeneration(
        f'Give me only the requirements of this job posting: {jobposting} from this json file and no extra preecding or trailing words other than the requirements')
    extrated_posting["skills"] = textgeneration(
        f'Give me only the skills of this job posting: {jobposting} from this json file and no extra preecding or trailing words other than the skills')
    extrated_posting["job description"] = textgeneration(
        f'Give me only the job description of this job posting: {jobposting} from this json file and no extra preecding or trailing words other than the job description')
    extrated_posting["responsibilities"] = textgeneration(
        f'Give me only the responsibilities of this job posting: {jobposting} from this json file and no extra preecding or trailing words other than the job description')
    extrated_posting["programming languages"] = textgeneration(
        f'Give me only the programming languages of this job posting: {jobposting} from this json file and no extra preecding or trailing words other than the programming languages')
    return extrated_posting

def cover_letter(extracted_resume,jobposting):
    prompt = f'generate me a cover letter based on the job postiong: {jobposting} and my resume: {extracted_resume}'
    content = textgeneration(prompt)
    return content

def pre_process2(extracted_resume,jobposting):
    with open("jakesresume.txt", 'r') as file:
        jresume = file.readlines()

    part1 = jresume[0:86]
    intro = jresume[87:99]
    education = jresume[100:111]
    experience = jresume[112:159]
    projects = jresume[159:181]
    techskill = jresume[183:189]

    with open("generate_text.txt",'w') as file:
        gen_desc = extracted_resume["GenDet"]
        prompt_gen = f"i have general details from my resume {gen_desc} use these details to replace the ALL details in {''.join(intro)} remove any trailing words or preceding words, this is part of a larger latex file so do not make new open close brackets or parentheses Do not use two subheadings under this field."
        content_gen = textgeneration(prompt_gen)

        file.write(''.join(part1))
        file.write('\n')
        file.write("%%%%%%%%%%%%%%%%%%%%%%%%%")
        file.write('\n')

        file.write(content_gen)
        file.write('\n')


        gen_edu = extracted_resume["Education"]
        prompt_edu = f"i have education details from my resume {gen_edu} use these details to replace ALL the details in {''.join(education)} and if a field is null remove it. remove any trailing words or preceding words, this is part of a larger latex file so do not make new open close brackets or parentheses. DO NOT REMOVE ANY FORMATTING SYMBOLS "
        content_edu = textgeneration(prompt_edu)

        file.write("%%%%%%%%%%%%%%%%%%%%%%%%%")
        file.write('\n')

        file.write(content_edu)
        file.write('\n')



        gen_exp = extracted_resume["Experience"]
        # prompt_exp = f"i have Experience details from my resume {gen_exp} .Fill up just that part for this part:\n {resume[2]}"
        prompt_exp = f'Here are my experiences from my resume that i made: {gen_exp} and here are the job requirements: {jobposting["requirements"]}and skills: {jobposting["skills"]}. Pick top 2 closest matching experiences from my given resume that match all details of the job and  make it match it in this given format: {"".join(experience)} remove any trailing words or preceding words, this is part of a larger latex file so do not make new open close brackets or parentheses. DO NOT REMOVE ANY FORMATTING SYMBOLS '
        content_exp = textgeneration(prompt_exp)

        file.write('\n')

        file.write("%%%%%%%%%%%%%%%%%%%%%%%%%")
        file.write('\n')

        file.write(content_exp)
        file.write('\n')
        gen_pro = extracted_resume["Projects"]
        # prompt_pro = f"i have Projects details from my resume {gen_pro} .Fill up just that part for this part:\n {resume[3]}"

        prompt_pro = f'Here are my projects from my resume that i made: {gen_pro} and here are the job requirements: {jobposting["requirements"]} and skills: {jobposting["skills"]} pick 2 projects that closely match the job description and make sure they are DIFFERENT FROM {content_exp}. make sure the projects are output in this format : {"".join(projects)}remove any trailing words or preceding words, this is part of a larger latex file so do not make new open close brackets or parentheses.DO NOT REMOVE ANY FORMATTING SYMBOLS '

        content_pro = textgeneration(prompt_pro)
        file.write('\n')

        file.write("%%%%%%%%%%%%%%%%%%%%%%%%%")
        file.write('\n')
        file.write(content_pro)
        file.write('\n')

        gen_skills = extracted_resume["TechSkills"]

        prompt_skills = f'Here are my skills that i know: {gen_skills} and here are the required skills for the job: {jobposting["skills"]} list the common skills and if the common skills are less than 3 then put important skills from the resume and do not put any extra preceding or trailing words and make it match this format exactly: {"".join(techskill)} remove any trailing words or preceding words, this is part of a larger latex file so do not make new open close brackets or parentheses.DO NOT REMOVE ANY FORMATTING SYMBOLS '

        content_skills = textgeneration(prompt_skills)
        #return content_skills
        file.write('\n')

        file.write("%%%%%%%%%%%%%%%%%%%%%%%%%")
        file.write('\n')

        file.write(content_skills)
        file.write('\n')
        file.write("\end{document}")


        print("Done Processing extract jobs!!!")

# def change_file_extension(file_path, new_extension):
#     # Get the directory and base filename without extension
#     directory, filename = os.path.split(file_path)
#     filename_without_extension, _ = os.path.splitext(filename)
#
#     # Create the new file path with the desired extension
#     new_file_path = os.path.join(directory, f"{filename_without_extension}.{new_extension}")
#
#     # Rename the file
#     os.rename(file_path, new_file_path)
#
#     return new_file_path

def latex_to_pdf(latex_file_path):
    try:
        # Run pdflatex command to compile LaTeX to PDF
        subprocess.run(["pdflatex", latex_file_path], check=True)
        print(f"PDF generated successfully: {latex_file_path.replace('.tex', '.pdf')}")
    except subprocess.CalledProcessError as e:
        print(f"Error during PDF generation: {e}")
    except Exception as ex:
        print(f"Unexpected error: {ex}")


def resumeGenerator(resume_location, website_url = "https://www.karkidi.com/job-details/43057-co-op-software-engineer-job"):
    jobposting = scrape_text_jobposting(website_url)
    # print(jobposting)

    jp = extract_jobdescription(jobposting)
    #print(jp)
    extracted_resume= extract_resume(resume_location)
    #print(extracted_resume)


    # preprocess2 is the actual resume generative command
    pre_process2(extracted_resume,jp)

    #result = generate_resume(resume,jp,extracted_resume)
    #print(resume)
    #conver_to_latex(resume,result)
    # define the file paths

    file_path = 'generate_text.txt'
    new_file_path = 'generate_text_latex.tex'

    # Delete the existing file
    if os.path.exists(new_file_path):
        os.remove(new_file_path)

    # Rename the file
    os.rename(file_path, new_file_path)

    tex_file_path = 'generate_text_latex.tex'

    output_pdf = 'output.pdf'
    
    latex_to_pdf(tex_file_path)

    # cover letter

def coverLetter(resume_location, website_url = "https://www.karkidi.com/job-details/43057-co-op-software-engineer-job"):
    jobposting = scrape_text_jobposting(website_url)
    # print(jobposting)

    jp = extract_jobdescription(jobposting)
    #print(jp)
    extracted_resume= extract_resume(resume_location)
    c_letter = cover_letter(extracted_resume,jobposting)
    print("Cover Letter: ",c_letter)
