from langchain.llms import OpenAI
import os
import requests
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

llm = OpenAI(openai_api_key=openai_api_key)
prompt_template = PromptTemplate.from_template("Generate a multiple choice questions about {topic} along with the correct answers, make sure that the question is mildly difficult, and make sure that the correct answer is labeled with 'Correct Answer:' and followed by the answer itself")

# Function to generate quiz questions for a given topic
def generate_quiz_questions(topic, num_questions=5, extra_questions=20):
    questions = []
    total_questions = num_questions + extra_questions

    for _ in range(total_questions):
        formatted_prompt = prompt_template.format(topic=topic)
        response = llm.invoke(formatted_prompt)
        #print("Raw response:", response)
        processed_responses = process_response(response)

        for processed in processed_responses:
            if isinstance(processed, dict) and len(processed.get('options', [])) == 4:
                questions.append(processed)
                if len(questions) == num_questions:
                    break  

        if len(questions) == num_questions:
            break  

    return questions

# Function to process the raw response from the language model
def process_response(response):
    quiz_data = []

    questions_blocks = response.strip().split("Q:")
    for block in questions_blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue

        question = lines[0].strip()

        correct_answer = None
        options = []
        for line in lines[1:]:
            if "Correct Answer:" in line:
                correct_answer = line.split("Correct Answer:")[1].strip()
            elif line.strip() and line[0].isalpha() and line[1] in [".", ")"]:
                option = line.strip()
                if "Correct Answer:" in option:  # Handling cases where the correct answer is on the same line
                    option, correct_answer = option.split("Correct Answer:")
                    correct_answer = correct_answer.strip()
                options.append(option.strip())

        if question and options and correct_answer:
            question_data = {
                "question": question,
                "options": options,
                "correct_answer": correct_answer
            }
            quiz_data.append(question_data)

    return quiz_data

# Code used for testing the script directly

# if __name__ == "__main__":
#     topic = "Animal Kingdom"
#     questions = generate_quiz_questions(topic, 5)
#     print(questions)