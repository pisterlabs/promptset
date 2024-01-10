import openai
import json
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from retry import retry
from tqdm import tqdm

# Initialize Firebase
cred = credentials.Certificate("cfa-creds.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Initialize OpenAI API
MODEL_NAME = "gpt-4"
TEMPERATURE = 0.2
PROMPT = """
Please classify the parsed question into a subseet of the most applicablee learning objectives found in the json object.
Your response must be a json object with the following format:
{{
    "predicted_objectives": [objective1, objective2, ...]
}}

EXAMPLE 1:
INPUT: {{
    "content": "2. The nominal risk-free rate is best described as the sum of the real risk-free rate \nand a premium for:\nA. maturity.\nB. liquidity.\nC. expected inflation.",
    "learning_objectives": [
        "Interpret interest rates as required rates of return, discount rates, or opportunity costs",
        "Explain an interest rate as the sum of a real risk-free rate and premiums that compensate investors for bearing distinct types of risk",
        "Calculate and interpret the future value (FV) and present value (PV) of a single sum of money, an ordinary annuity, an annuity due, a perpetuity (PV only), and a series of unequal cash flows",
        "Demonstrate the use of a time line in modeling and solving time value of money problems",
        "Calculate the solution for time value of money problems with different frequencies of compounding",
        "Calculate and interpret the effective annual rate, given the stated annual interest rate and the frequency of compounding"
    ],
    "learning_module_name": "Learning Module 1\tThe Time Value of Money",
    "parsed_question": {{
        "id": "2",
        "question": "The nominal risk-free rate is best described as the sum of the real risk-free rate and a premium for:",
        "options": {{
            "A": "maturity",
            "B": "liquidity",
            "C": "expected inflation"
        }},
        "data": ""
    }}
}}
ANSWER: {{"predicted_objectives": ["Interpret interest rates as required rates of return, discount rates, or opportunity costs",
    "Explain an interest rate as the sum of a real risk-free rate and premiums that compensate investors for bearing distinct types of risk"]}}

INPUT:{}
ANSWER:"""


@retry(tries=5, delay=2, backoff=2)  # delay and backoff are in seconds
def classify_question(question_data):
    # Send a request to the OpenAI API to classify the question
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful AI."},
            {"role": "user", "content": PROMPT.format(question_data)}
        ],
        temperature=TEMPERATURE
    )
    # Extract the generated text from the response
    generated_text = json.loads(
        response['choices'][0]['message']['content'])

    return generated_text


def main():
    # Get the questions from the Firebase collection
    questions_ref = db.collection('questions')
    page_size = 200  # Set the number of documents to retrieve per page
    last_doc = None

    while True:
        if last_doc:
            questions = questions_ref.order_by(
                'parsed_question.id', direction=firestore.Query.ASCENDING
            ).start_after(last_doc).limit(page_size).stream()
        else:
            questions = questions_ref.order_by(
                'parsed_question.id', direction=firestore.Query.ASCENDING
            ).limit(page_size).stream()
        questions_list = list(questions)

        if not questions_list:
            break

        # Loop over each question
        for question in tqdm(questions_list, desc="Classifying questions"):
            # Get the question data
            question_data = question.to_dict()

            if 'predicted_objectives' in question_data.keys():
                print("Skipping question that has already been classified")
                continue

            # Send a request to the OpenAI API to classify the question
            response = classify_question(question_data)

            # Insert the predicted objectives back into the Firebase database
            doc_ref = db.collection('questions').document(question.id)

            try:
                doc_ref.update(response)
            except Exception as e:
                print(f"Failed to update Firestore document {doc_ref.id}: {e}")

            last_doc = questions_list[-1]

        if len(questions_list) < page_size:
            break  # Exit the loop if there are no more documents


if __name__ == "__main__":
    main()
