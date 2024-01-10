import openai
import numpy as np

# Step 1: Data Collection
Pq = ["What is 1 + 1?", "What is 5 - 2?", "What is 3 * 4?", "What is 10 / 2?"]
Pc = ["Calculate: 1 + 1", "Calculate: 5 - 2", "Calculate: 3 * 4", "Calculate: 10 / 2"]

# Step 2: Model Testing
def get_responses(prompts):
    responses = []
    for prompt in prompts:
        response = openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=5)
        responses.append(response.choices[0].text.strip())
    return responses

Rq = get_responses(Pq)
Rc = get_responses(Pc)

# Step 3: Response Evaluation
def D(r, correct_answers):
    if r in correct_answers:
        return 1
    elif any(correct_answer in r for correct_answer in correct_answers):
        return 0.5
    else:
        return 0

correct_answers = ["2", "3", "12", "5"]
Dq = [D(r, correct_answers) for r in Rq]
Dc = [D(r, correct_answers) for r in Rc]

# Step 4: Data Analysis
average_Dq = np.mean(Dq)
average_Dc = np.mean(Dc)

# Step 5: Hypothesis Verification
if average_Dc > average_Dq:
    print("Hypothesis is verified.")
else:
    print("Hypothesis is not verified.")