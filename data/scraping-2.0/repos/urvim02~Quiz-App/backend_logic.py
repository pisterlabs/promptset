import openai
import langchain

openai.api_key = "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"

def generate_question(prompt):
    response = openai.Completion.create(
        engine="davinci-codex",
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].text.strip()

def generate_answer_options(question, prompt):
    response = openai.Completion.create(
        engine="davinci-codex",
        prompt=prompt,
        max_tokens=50,
        n=3,
        stop=None,
        temperature=0.5,
    )
    return [option.text.strip() for option in response.choices]

def generate_quiz(topic, number_of_questions):
    quiz_data = []
    for i in range(number_of_questions):
        question_prompt = f"Generate a question related to {topic}.\n{topic}"
        question = generate_question(question_prompt)

        answer_options_prompt = f"Provide 3 answer options for the question: {question}.\n{question}"
        answer_options = generate_answer_options(question, answer_options_prompt)

        correct_answer_prompt = f"Select the correct answer for the question: {question}. The options are: {', '.join(answer_options)}.\n{question}"
        correct_answer = openai.Completion.create(
            engine="davinci-codex",
            prompt=correct_answer_prompt,
            max_tokens=50,
            n=1,
            stop=None,
            temperature=0.5,
        ).choices[0].text.strip()

        quiz_data.append((question, answer_options, correct_answer))
    return quiz_data