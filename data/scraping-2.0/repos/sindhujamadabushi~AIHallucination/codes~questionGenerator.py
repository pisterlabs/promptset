import json
from openai import OpenAI
import os

client = OpenAI(api_key="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")

with open("../inputs/contexts.txt", "r") as f:
    contexts = f.readlines()

def generate_questions(context, question_type, num_questions_per_context_qtype):
    questions = []
    prompt_count = f"Based on the following context, generate {num_questions_per_context_qtype} 'count' questions. A 'count' question's answer is always a number. A 'count' question's answer is NOT a date or a year: \\n{context}."
    prompt_yesno = f"Based on the following context, generate {num_questions_per_context_qtype} 'yesno' questions. A 'yesno' question's answer is either 'yes' or 'no': \\n{context}"

    if question_type == 'count':
        response = client.completions.create(
        model="text-davinci-002",
        prompt = prompt_count,
        max_tokens=250)
        output = response.choices[0].text.strip().split("\\n")
        for question in output[:num_questions_per_context_qtype]:
                if question.strip():  
                    questions.append(question.strip())
    if question_type == 'yesno':
        # return ['a\nb']
        response = client.completions.create(
        model="text-davinci-002",
        prompt = prompt_yesno,
        max_tokens=250)
        output = response.choices[0].text.strip().split("\\n")
        for question in output[:num_questions_per_context_qtype]:
                if question.strip():  
                    questions.append(question.strip())
    return questions

num_contexts = 5
num_questions_per_context_qtype = 5
# all_contexts = contexts[:num_contexts]

directory_name = "../results/seedQA"
question_types = ["count", "yesno"]

for c in range(21, len(contexts)):
    context_questions = {"context": contexts[c], "count": [], "yesno": []}
    for q_t in question_types:
        questions = generate_questions(contexts[c], q_t, num_questions_per_context_qtype)
        questions = questions[0].split('\n')
        context_questions[q_t] = questions
    
    # Create a directory for questions
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    # Create a file for each context
    with open(directory_name + '/' + f"context_{c}.json", "w") as f:
        json.dump(context_questions, f)  