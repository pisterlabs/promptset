import random
import nltk
from scipy import stats
import openai

# 1. Data Collection
questions = ["What is 1 + 1?", "What is the capital of France?", "Who wrote 'To Kill a Mockingbird'?", "What is the chemical symbol for water?"]
modified_questions = ["Provide a one-word answer: What is 1 + 1?", "Provide a one-word answer: What is the capital of France?", "Provide a one-word answer: Who wrote 'To Kill a Mockingbird'?", "Provide a one-word answer: What is the chemical symbol for water?"]

# 2. Experiment Setup
random.shuffle(questions)
random.shuffle(modified_questions)

# 3. Execution
def get_gpt3_response(prompt):
    response = openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=100)
    return response.choices[0].text.strip()

original_responses = [get_gpt3_response(q) for q in questions]
modified_responses = [get_gpt3_response(q) for q in modified_questions]

# 4. Data Analysis
def calculate_metrics(responses):
    lengths = [len(nltk.word_tokenize(r)) for r in responses]
    complexities = [len(nltk.sent_tokenize(r)) for r in responses]
    return sum(lengths) / len(lengths), sum(complexities) / len(complexities)

L1, C1 = calculate_metrics(original_responses)
L2, C2 = calculate_metrics(modified_responses)

# 5. Hypothesis Testing
length_p_value = stats.ttest_ind([len(nltk.word_tokenize(r)) for r in original_responses], [len(nltk.word_tokenize(r)) for r in modified_responses]).pvalue
complexity_p_value = stats.ttest_ind([len(nltk.sent_tokenize(r)) for r in original_responses], [len(nltk.sent_tokenize(r)) for r in modified_responses]).pvalue

# 6. Conclusion
if length_p_value < 0.05 and complexity_p_value < 0.05:
    conclusion = "The modification of the prompting strategy results in more concise responses from the LLM."
else:
    conclusion = "The modification of the prompting strategy does not significantly affect the conciseness of the LLM's responses."

# 7. Reporting
report = f"""
Data Collection:
Original Questions: {questions}
Modified Questions: {modified_questions}

Experiment Setup:
Original Prompts: {questions}
Modified Prompts: {modified_questions}

Execution:
Original Responses: {original_responses}
Modified Responses: {modified_responses}

Data Analysis:
Average Length and Complexity for Original Prompts: {L1, C1}
Average Length and Complexity for Modified Prompts: {L2, C2}

Hypothesis Testing:
P-value for Length: {length_p_value}
P-value for Complexity: {complexity_p_value}

Conclusion:
{conclusion}
"""
print(report)