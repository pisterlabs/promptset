import os
import sys
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_posting(file):
    co0p_job = dict()

    with open(file, 'r') as f:
        text = f.read()
        try:
            info, desc = text.split('Job Description:')
        except:
            info1, info2, desc = text.split('Job Description:')
            info = info1 + info2
        info = info.split(':')
        
        co0p_job['org'] = text.split('Organization Name')[-1].split('\n')[0]

        new_info = []
        for i in range(1, len(info)):
            info[i] = info[i].split('\n')
            try:
                new_info.append(info[i][-1] + info[i+1].split('\n')[0])
            except:
                pass

        if desc[0] == ' ':
            desc = desc[1:]
        try:
            desc = desc.split('We thank all candidates who apply')[0]
        except:
            desc = desc.split('Please be sure to click')[0]
        co0p_job.update({'info': new_info, 'description': desc})

        return co0p_job
    
def generate_customized_cover_letter(job_description, max_tokens=1600, model="gpt-4"):
    '''
    Returns customized cover letter for the job description.
    '''

    with open("base_cover_letter.txt", 'r') as f:
        base_cover_letter = f.read()
    
    prompt = f"Create a custom cover letter for a job posting.\n\n{base_cover_letter}\n\nJob Description:\n{job_description}"
    print(prompt)

    appr_input_token_count = len(prompt.split(' '))

    # Price confirmation
    if model == "gpt-4-1106-preview":
        input_per_1k_tokens = 0.001
        output_per_1k_tokens = 0.003
    elif model == "gpt-4":
        input_per_1k_tokens = 0.06
        output_per_1k_tokens = 0.12

    guess_output_token_count = int(appr_input_token_count * 1.5)

    total_input_cost = input_per_1k_tokens * (appr_input_token_count / 1000)
    total_output_cost = output_per_1k_tokens * (guess_output_token_count / 1000)
    max_cost = total_input_cost + total_output_cost

    print('\nUsing model: ' + model)
    print('NOT ACCURATE PRICING: The max cost for the request is ' + str(max_cost) + '$.')
    user_response = input(f"NOT ACCURATE PRICING: The expected cost for the request is ${total_output_cost+total_input_cost:.2f}. \n\nDo you wish to continue? (yes/no): ")
    print("-"*50)
    # user_response = 'yes'
    
    if user_response.lower() == 'yes' or user_response.lower() == 'y':
        # Your code to make the API request
        response = client.chat.completions.create(model=model,
        messages=[
                    {"role": "system", "content": 'Please create a custom cover letter for a job posting. I will provide some information about myself and the cover letter specification, then the job posting.'}, 
                    {"role": "user", "content": prompt}
                  ],
        max_tokens=max_tokens)
        return response.choices[0].message.content.strip()
    else:
        print(base_cover_letter)
        print("-"*50)
        print(job_description)
        print("Operation cancelled by the user.")
        return None

def process_posting(filename):
    job_description_path = os.path.join("postings", f"{filename}.txt")
    print(job_description_path)
    if os.path.exists(job_description_path):
        coop_job = get_posting(job_description_path)
        job_description = coop_job['org'] + ' '.join(coop_job['info']) + coop_job['description']
        print(f"Job description for {filename}:\n{job_description}\n{'-'*50}")
        customized_cl = generate_customized_cover_letter(job_description, model='gpt-4-1106-preview')
        
        if customized_cl is None:
            return
        
        print(f"Saving cover Letter for {filename}:\n{customized_cl}\n{'-'*50}")
        with open(f"modified_cover_letters/CL_{filename}.txt", 'w') as f:
            f.write(customized_cl)
        print("Cover letter saved successfully.\n")
    else:
        print(f"File {filename} not found in the 'postings' directory.")

def main():
    if len(sys.argv) < 2:
        print("Please provide job IDs as arguments.")
        return
    
    job_ids = sys.argv[1:]
    for job_id in job_ids:
        process_posting(job_id)

if __name__ == "__main__":
    main()