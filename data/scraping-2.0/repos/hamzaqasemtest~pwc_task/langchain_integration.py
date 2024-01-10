from langchain.llms import OpenAI
import json


llm = OpenAI(api_key="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")

def generate_questions(topic, num_questions):

    questions = []
    for _ in range(num_questions):

        prompt = f"Create a multiple-choice question about {topic} readable for the user. with the correct answer in json like question: , choices: [answer1,answer2,answer3], answer:answer1"
        response = llm.generate([prompt])  # Pass the prompt inside a list
        question = parse_response_into_question(response)
        questions.append(question)

    return questions

def parse_response_into_question(response):

        response_string = response.generations[0][0].text
        cleaned_string = response_string.strip()
        corrected_string = cleaned_string.replace('question:', '"question":').replace('choices:', '"choices":').replace('answer:', '"answer":')
        json_object = json.loads(corrected_string)

        parsed_question = {
            "question": json_object["question"],
            "choices": json_object["choices"],
            "answer": json_object["answer"]
        }

        return parsed_question


