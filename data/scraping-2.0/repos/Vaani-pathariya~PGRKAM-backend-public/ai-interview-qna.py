import openai
import difflib

# Set your OpenAI API key
openai.api_key 

# Function to generate answers to questions
def generate_answer(question):
    prompt = f"Question: {question}\nAnswer:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
        stop=None
    )
    answer = response['choices'][0]['text'].strip()
    return answer

# Function to check if the user's answer is correct
def is_answer_correct(user_answer, correct_answer):
    # Use difflib to get a similarity ratio
    similarity_ratio = difflib.SequenceMatcher(None, user_answer.lower(), correct_answer.lower()).ratio()

    # You can adjust the threshold based on your needs
    similarity_threshold = 0.7

    return similarity_ratio >= similarity_threshold

# Example usage
question_to_ask = """
I think........ , I am not very sure , but the answer to that question might be
Analyze the confidence(literal) and give only percentage from the overall text"""
correct_answer = generate_answer(question_to_ask)

user_answer = input(f"Question: {question_to_ask}\nEnter your answer: ")

if is_answer_correct(user_answer, correct_answer):
    print("Correct!")
else:
    print(f"Incorrect. The correct answer is: {correct_answer}")
