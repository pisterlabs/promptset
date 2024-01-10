import os
import openai
import pandas as pd

# Set up OpenAI API key
openai.api_key = os.getenv("OpenAI API key")
openai.api_key = 'OpenAI API key'

def grade_response(response_text):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system", 
                "content": "You will grade the response basedon its alignment with feedback best practices from the provided research. \nThe grading criteria overall is: \n\n1) Explain how to effectively respond to a student who has just made an error.\n2) Apply recommended strategies, for responding to students when they make an error, that increase motivation and engagement.\n\nThe grading criteria per question is: \n\nQuestion 1: \"What exactly would you say to Jedidiah regarding his mistake, to effectively respond in a way that increases his motivation to learn?\" \nCriteria for Q1: Student should not point out the mistake and should praise the student's effort.\nQuestion 2: \"Why do you think the tutors response you selected in Question 10 will best support Jedidahs motivation to learn and increase engagement?\" \nCriteria for Q2: Students should mention that not pointing out the mistake allows students to self-correct, which increases learning and motivation.\n\nResearch Summary: Studies have shown that the way tutors intervene or respond when students make mistakes or show misconceptions in their learning can affect the students motivation to learn. The best approach is to ask students to try and correct their own mistakes, respond indirectly without directly pointing out errors, recognize 'easy mistakes' on their own, develop critical thinking skills, involve students in the learning process, avoid causing too much frustration, and praise students for effort. Based on this, grade the response as either 0 or 1."
            }, 
            {
                "role": "user", 
                "content": response_text
            }
        ]
    )
    
    score = completion.choices[0].message["content"]
    
    if "1" in score:
        return 1
    elif "0.5" in score:
        return 0.5
    else:
        return 0


# Load the CSV file
data = pd.read_csv('ungraded_single_responses copy.csv')

# # Process only the first 10 rows
# data = data.head(10)

# # Add a new 'score' column
# data['score'] = data['response'].apply(grade_response)
# Define the batch size
BATCH_SIZE = 10

# Create an empty list to store all scores
all_scores = []

# Loop through the data in batches
for i in range(0, len(data), BATCH_SIZE):
    batch = data.iloc[i:i+BATCH_SIZE]
    batch_scores = batch['response'].apply(grade_response).tolist()
    all_scores.extend(batch_scores)

# Add the scores to the dataframe
data['score'] = all_scores

# Save the dataframe with scores back to CSV
data.to_csv('responses_with_scores_final_v2.csv', index=False)