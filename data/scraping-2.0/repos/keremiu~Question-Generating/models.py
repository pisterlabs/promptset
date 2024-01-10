import openai
import pandas as pd
import numpy as np
import re
openai.api_key = "your_api_key"

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
        print(question)
        print(question[0])
        print(question[1])
        answer_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"What is the correct answer for: {question}"}
            ]
        )
        answer = answer_response.choices[0].message['content'].strip()

        # Answer often starts with "The correct answer is". Let's handle that.
        #if answer.startswith("The correct answer is"):
        #    answer = answer.split(".")[1].strip()

        # Use regex to split by options
        question = generated_content[0]
        options = generated_content[1]

        print(generated_content)
        parts = re.split(r"\n([A-Da-d])\)", str(generated_content))
        print(str("partların hepsi")+str(parts))
        options = [parts[i] + parts[i+1] for i in range(1, len(parts)-1, 2)]

        extra_option = [opt for opt in options if answer not in opt]

        questions.append(main_question)
        answers.append(answer)
        extra_options.append(", ".join(extra_option))

        print(f"Question {i + 1}:\n{question}\n")

    return questions, extra_options, answers,prompt,generated_content
  
def generate_questions_for_all_topics(difficulty="easy",n=1):
    all_questions, all_extra_options, all_answers = [], [], []
    dump = ""
    # Her ana konu için döngü başlat
    for main_topic, sub_topics in topics.items():
        print(f"Generating questions for {main_topic}...")
        
        for sub_topic in sub_topics:
            questions, extra_options, answers,dump,generated_content = generate_questions(difficulty=difficulty, topic=sub_topic, n=n)  # Burada 'medium' zorluk seviyesi seçildi, ihtiyaca göre değiştirilebilir.
            print(dump)
            print("-----------------")
            print(questions)
            print("-----------------")
            print(extra_options)
            print("-----------------")
            print(answers)
            print("-----------------")
            print(generated_content)
            all_questions.extend(questions)
            all_extra_options.extend(extra_options)
            all_answers.extend(answers)
    
    # Üretilen soruları bir DataFrame'e dönüştür
    data = {'Question': all_questions, 'Extra Options': all_extra_options, 'Answer': all_answers, 'Main Topic': [main_topic for main_topic, sub_topics in topics.items() for _ in sub_topics], 'Sub Topic': [sub_topic for _, sub_topics in topics.items() for sub_topic in sub_topics]}
    df = pd.DataFrame(data)
    
    return df
  
def user_selection():
    print("Lütfen bir seçenek belirtin:")
    print("1: Tüm konular için soru üret")
    print("2: Belirlediğiniz bir konu için soru üret")
    choice = int(input("Seçiminizi yapın (1/2): "))

    difficulty = input("Lütfen zorluk seçin (easy/medium/hard): ").lower()
    n = int(input("Bir seferde kaç soru üretmek istiyorsunuz? "))

    if choice == 1:
        df = generate_questions_for_all_topics(difficulty, n)
        filename = 'generated_mcq_questions_all_topics.csv'
        df.to_csv(filename, index=False)
        print(f"Sorular {filename} dosyasına kaydedildi.")

    elif choice == 2:
        topic = input("Özel bir Python konusu girin (örn. 'class', 'args') veya genel için boş bırakın: ")
        questions, extra_options, answers = generate_questions(difficulty, topic, n)

        # Üretilen soruları bir DataFrame'e dönüştür
        data = {'Question': questions, 'Extra Options': extra_options, 'Answer': answers}
        df = pd.DataFrame(data)

        # DataFrame'i CSV dosyasına kaydet
        filename = f'generated_mcq_questions_{difficulty}.csv'
        df.to_csv(filename, index=False)
        print(f"Sorular {filename} dosyasına kaydedildi.")

    else:
        print("Geçersiz seçenek!")
        user_selection()
user_selection()        
