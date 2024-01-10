from openai import OpenAI
from credentials import chatgpt_api


client = OpenAI(api_key=chatgpt_api)


def analyze_job_fit(job_description, resume):

    response = client.chat.completions.create(model="gpt-3.5-turbo",  # Replace with the correct model name
    messages=[
        {"role": "user", "content": f"""
        I am trying to apply to a job. This is my resume: {resume}
        
        THIS IS THE JOB DESCRIPTION: {job_description}
        
        Do you think this job is a good fit for me? What is your confidence that I would be able to get this job? Give your answer in JSON format, in which contains 2 fields:
        ConfidenceRating, a numerical value from 0-100 that is your percent in confidence that I would get the job. If the job has several requirements that my resume does not meet, especially considering technologies I have little to no experience in, the value should be close to 0. However, if my qualifications match the job well, the value should be close to 100.
        RequirementsAnalysis, a string value that holds your analysis of the requirements compared to my resume, acting as a reasoning for your ConfidenceRating.
        """
         }
    ])

    return response.choices[0].message.content

