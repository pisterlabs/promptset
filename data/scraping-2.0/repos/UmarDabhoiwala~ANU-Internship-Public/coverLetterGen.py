from halo import Halo
import openai
import config 
from PyPDF2 import PdfReader
import pypandoc

spinner = Halo(text='Loading', spinner='dots')
openai.api_key = config.OPENAI_API_KEY

def cover_letter_gen(file_path, company_name, job_title):
    
    print(file_path, company_name, job_title)
    
    resume_text = ""
    
    try:
        if file_path == "files/input_resume.pdf":
            reader = PdfReader("files/input_resume.pdf")
            for x in range(0, len(reader.pages)):
                page = reader.pages[x]
                the_text = page.extract_text()
                resume_text += the_text
        elif file_path == "files/input_resume.docx":
            resume_text = pypandoc.convert_file(file_path, "plain")
        else: 
            raise Exception("Invalid file type")
    except:
        print("bad file type")


    init_prompt = """You are a cover letter writer. The user will input a plain text version of thier resume, 
    the company and the title that they are applying to. You will then generate a cover letter for them. 
    For the writing style keep it formal and professional but not overly verbose or boring. 
    Give your response in markdown format and only give the answer."""

    prompt = f"""This is the resume in plain text {resume_text}. 
    The name of the company they are applying to is {company_name}
    The title they are applying to is {job_title}"""


    spinner.start("Text Generating")

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "system", "content": init_prompt},
            {"role": "user", "content": prompt}]
        )

    spinner.succeed("Generation Complete")

    response = completion['choices'][0]["message"]["content"]

    #Writing prompt
    with open(f'files/coverletter.md', 'w') as f:
        f.write(response)


    reference_docx = 'files/reference.docx'
    pandoc_args = [
        '--reference-doc=' + reference_docx,
        ]
    
    output = pypandoc.convert_file(f'files/coverletter.md', 'docx', outputfile=f'files/coverletter.docx',extra_args=pandoc_args)

