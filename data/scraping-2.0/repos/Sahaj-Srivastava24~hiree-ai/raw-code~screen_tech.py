import json
import openai
import random

# Set your OpenAI API key
openai.api_key = "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"

# Load candidate info from JSON
with open("E:/hackathon/info_demo.json", 'r') as json_file:
    data = json.load(json_file)
    candidate = data  # Assuming the JSON structure matches the provided data

# Define a sample job description
questions_about=data["work_experience"]
candidate_tech=data["Skills"]
job_description = "Frontend Web developer"
jd_tech_keywords = ["react","frontend","javascript"]

# Extract tech stack from candidate's project experience
tech_stack = "good react developer"

# Combine JD and tech stack keywords for generating questions
keywords = jd_tech_keywords + tech_stack.split()

# Generate questions using OpenAI GPT-3
prompt2=f"generate 5 technical questions based on common techstack between {job_description} and {candidate_tech}"

response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt2,
    max_tokens=2000,
    stop=None,
    temperature=0.7
)

generated_question = response.choices[0].text.strip()

print(f"Generated question: {generated_question}")


# # Load questions from the JSON file
# with open('./questions.json', 'r') as questions_file:
#     tech_questions = json.load(questions_file)

# # Check if the generated question is in the tech_questions
# for tech_keyword in jd_tech_keywords:
#     if tech_keyword in tech_questions and any(q['question'] == generated_question for q in tech_questions[tech_keyword]):
#         print("Question:", generated_question)
#         break
# else:
#     print("No relevant questions found for this tech stack.")
