import openai
import os

# Set up the OpenAI API key
openai.api_key = "your_openai_api_key"

def read_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def get_suggestions(resume, job_description):
    prompt = f"Given the following resume:\n{resume}\nAnd the following job description:\n{job_description}\nPlease suggest edits to improve the resume to match the job requirements."

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.7,
    )

    return response.choices[0].text.strip()

def main():
    resume_path = "path/to/your/resume.txt"
    job_description_path = "path/to/job_description.txt"

    resume = read_file(resume_path)
    job_description = read_file(job_description_path)

    suggestions = get_suggestions(resume, job_description)
    print("Suggested edits for your resume:")
    print(suggestions)

if __name__ == "__main__":
    main()