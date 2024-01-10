import openai
import pandas as pd

openai.api_key = "your_api_key"

def generate_questions(difficulty, topic="", n=1):
    base_prompts = {
        "easy": "easy Python-related",
        "medium": "medium difficulty Python-related",
        "hard": "hard Python-related"
    }
    
    if topic:
        topic_prompt = f" about {topic}"
    else:
        topic_prompt = ""

    questions, extra_options, answers = [], [], []

    prompt = f"Generate {n} {base_prompts[difficulty]} multiple choice questions{topic_prompt}."

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.8,
        max_tokens=500,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    generated_content = response.choices[0].message['content'].split("\n\n")

    for i, question in enumerate(generated_content):
        if i >= n:
            break

        answer_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"What is the correct answer for: {question}"}
            ]
        )
        answer = answer_response.choices[0].message['content'].strip()
        
        options = question.split("\n")[1:]
        extra_option = [opt for opt in options if answer not in opt]

        questions.append(question)
        answers.append(answer)
        extra_options.append(", ".join(extra_option))

    return questions, extra_options, answers

difficulty = input("Please select difficulty (easy/medium/hard): ").lower()
n = int(input("How many questions do you want to generate for each sub-topic? "))

topics = {
    "Basic Information": ["Python history and features", "Installation and environment setup", "Variables and data types", "Simple input/output operations"],
    "Control Structures": ["Conditional statements (if, elif, else)", "Loops (for, while)", "Flow control (break, continue, pass)"],
    "Functions": ["Function definition and calling", "Parameters and return values", "Anonymous functions (lambda)", "Local and global variables"],
    "Data Structures": ["Lists", "Tuples", "Dictionaries", "Sets"],
    "Object Oriented Programming (OOP)": ["Class definitions", "Objects and instances", "Inheritance", "Polymorphism", "Encapsulation"],
    "Exception Handling": ["Try, except blocks", "Types of exceptions", "Defining your own exceptions"],
    "File Operations": ["File open, read, write", "File locations and paths", "File related errors"],
    "Modules and Packages": ["Built-in modules", "Creating your own modules", "Package usage and creation"],
    "Basic Libraries": ["Standard libraries like os, sys, datetime, math", "Third-party libraries like requests, beautifulsoup4"],
    "Database Operations": ["Connecting with databases like SQLite, MySQL, PostgreSQL", "SQL queries and operations"],
    "Web Development": ["Frameworks like Flask, Django", "RESTful services"],
    "Data Science and Machine Learning": ["Data processing libraries like NumPy, Pandas", "Machine learning libraries like scikit-learn, TensorFlow, PyTorch"],
    "Other Topics": ["Creating GUI (like PyQt, Tkinter)", "Asynchronous programming (asyncio)", "Advanced topics like decorators, generators"]
}

all_questions, all_extra_options, all_answers, all_topics, all_sub_topics = [], [], [], [], []

for topic, subtopics in topics.items():
    for sub_topic in subtopics:
        print(f"Generating questions for: {sub_topic}")
        questions, extra_options, answers = generate_questions(difficulty, sub_topic, n)
        all_questions.extend(questions)
        all_extra_options.extend(extra_options)
        all_answers.extend(answers)
        all_topics.extend([topic] * len(questions))
        all_sub_topics.extend([sub_topic] * len(questions))

data = {
    'Topic': all_topics,
    'Sub-Topic': all_sub_topics,
    'Question': all_questions,
    'Extra Options': all_extra_options,
    'Answer': all_answers
}

df = pd.DataFrame(data)
filename = f'generated_mcq_questions_{difficulty}.csv'
df.to_csv(filename, index=False)
print(f"All questions saved to {filename}")
