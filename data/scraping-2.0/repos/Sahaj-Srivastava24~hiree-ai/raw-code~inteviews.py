import json
import openai

# Set your OpenAI API key
openai.api_key = "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"

# Load job description, screening questions, and candidate data
with open("job_description.json", 'r') as job_desc_file:
    job_description = json.load(job_desc_file)



with open("info_demo.json", 'r') as candidate_data_file:
    candidate_data = json.load(candidate_data_file)

JD=job_description["job_description"]



# Create a function to conduct the first-round interview
def conduct_first_round_interview(candidate_data, screening_questions, job_description):
    # Loop through each candidate
    for candidate in candidate_data:
        print(f"Interviewing candidate: {candidate['full_name']}")
        
        total_score = 0
        # Ask each screening question
        for question_key in screening_questions:
            question = screening_questions[question_key]["question"]
            importance = question_importance[question_key]
            
            # Generate a prompt for the question
            prompt = f"Candidate: {candidate['full_name']}\nQuestion: {question}\nAnswer:"
            
            # Generate an answer using OpenAI GPT-3
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=100,  # Adjust as needed
                stop=None,
                temperature=0.7
            )
            answer = response.choices[0].text.strip()
            
            # Evaluate answer based on importance level
            if importance == "high":
                score = len(answer.split())  # You can implement a more sophisticated scoring mechanism
            elif importance == "medium":
                score = len(answer.split()) // 2
            else:
                score = len(answer.split()) // 4
            
            total_score += score
            print(f"Question: {question}\nAnswer: {answer}\nScore: {score}\n")
        
        # Determine if candidate proceeds to the next round based on total score
        if total_score >= job_description['minimum_required_score']:
            print(f"Candidate {candidate['full_name']} proceeds to the next round!\n")
        else:
            print(f"Candidate {candidate['full_name']} did not meet the required score.\n")

# Call the function to conduct first-round interviews
conduct_first_round_interview(candidate_data, screening_questions, job_description)
