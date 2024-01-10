import openai
from dotenv import load_dotenv
import os

# Load the environment variables from the .env file
load_dotenv()

# Access the API key
api_key = os.getenv("API_KEY")

openai.api_key = api_key

def check_answer(Teachers_solution, Students_answer, Max_marks, Question):
    openai.api_key = api_key
    # openai.api_key = api_key_input
    # try:
    print("sending to gpt3")
    completion1 = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        
           

       messages = [
    {
        "role": "system",
        "content": "You are a strict teacher evaluating student answers.",
    },
    {
        "role": "user",
        "content": f'''Please evaluate the student's answer for the following question. You will be provided with the teacher's solution, the question, the student's answer, and the maximum marks. Your task is to assign a score to the student's answer.

**Teacher's Solution:**
{Teachers_solution}

**Question:**
{Question}

**Student's Answer:**
{Students_answer}

**Max Marks:**
{Max_marks}
**Important stuff** 
- Make sure to deduct marks wherever you can ( you have to be really strict)
- Make sure to give the response in the specified format
**Evaluation Criteria:**
- Accuracy: Compare the student's answer to the teacher's solution. Deduct 0.5 marks for each factual inaccuracy.
- Completeness: Consider the depth of coverage in the student's answer. Deduct 0.5 marks for each missing key point.
- Relevance: Assess if the student's answer stays on-topic. Deduct 0.5 marks for each irrelevant point.
- Clarity: Evaluate the clarity and organization of the student's response. Deduct 0.5 marks for incoherent or poorly structured answers.

**Marks Allocation:**
- Full Marks: Give full marks (as specified) for answers that match the teacher's solution exactly(context and accuracy wise).
- Partial Marks: Deduct 1 marks for any discrepancies between the student's answer and the teacher's solution, applying a clear grading scale.
- Length: If the student's answer is significantly shorter or longer than the teacher's solution, adjust the marks accordingly according to the content.(too short -3 marks ,short -2 marks, little short -1 marks)
- Explaination: If the student's answer doesnt contain the explaination of the answer that is there in the teachers answer deduct 0.5 marks.
You should consider all evaluation criteria and allocate marks based on the provided guidelines and just return the total marks allocated out of max marks.

YOU HAVE TO GIVE THE RESPONSE IN THIS FORMAT : {{ "marks": int,"explaination": string,"accuracy": string,"completeness":int(marks) ,"relevance": int,"clarity": int }} make sure you follow the format and give just integer values where asked and string where asked 
all the features accuracy , completeness,relavance,clarity,length should be positive integers ( the number of marks to be deducted ) 
'''
    }
],

# Your code to interact with the model here

        temperature=1,
        # max_tokens=15000,
    )
        
    final_html = completion1['choices'][0]['message']['content']

    return final_html




