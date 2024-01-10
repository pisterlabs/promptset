import os
import openai


openai.api_key = os.getenv("OPENAI_API_KEY")

# Read the job description from the file
with open("jobDescription.txt", "r") as file:
    job_description = file.read()

# Read the user's resume from the file
with open("resume.txt", "r") as file:
    resume = file.read()

nWords = 300

# Initialize the conversation with the assistant
conversation = [
            {"role": "system", "content": "You are a helpful assistant that writes cover letters."},
            {"role": "user", "content": f"My resume: {self.resume}"},
            {"role": "user", "content": f"Write a cover letter based on this job description: {self.job_description}"},
            {"role": "user", "content": f"Keep the length of the letter around {self.nWords} words."}
        ]

# Send the conversation to the API
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=conversation
)

# Extract the cover letter from the response
cover_letter = response['choices'][0]['message']['content']

# Save the cover letter to a file
with open("coverLetter.txt", "w") as file:
    file.write(cover_letter)