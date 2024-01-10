import openai
import csv

# Set your OpenAI API key
openai.api_key = "#hna yji lapi ta3na mana9adrouch nzidoh pusk yweli public fi github"

# Define the CSV file containing Stack Overflow questions
csv_file = "stackoverflow-answers-scraping/stackoverflow_questions.csv"
# Define the output text file
output_file = "Chat-GPT4-ANSWERS/ChatGPT4_answers.txt"

# Function to read Stack Overflow questions from the CSV file
def read_stackoverflow_questions(csv_file):
    questions = []
    with open(csv_file, "r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            questions.append(row[2])  # Assuming the question title is in the third column
    return questions

# Function to ask ChatGPT-4 and get 5 answers for each question
def ask_chatgpt4(questions):
    answers_per_question = {}
    for question in questions:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=f"Answer the following question:\n{question}\n\nAnswer:",
            max_tokens=100,  # Adjust as needed
            n=5,  # Get 5 answers per question
            stop=None,
            temperature=0.7,  # Adjust as needed
        )
        answers_per_question[question] = [choice.text.strip() for choice in response.choices]

    return answers_per_question

if __name__ == "__main__":
    # Read Stack Overflow questions from the CSV file
    stackoverflow_questions = read_stackoverflow_questions(csv_file)

    # Ask ChatGPT-4 and get 5 answers for each question
    answers = ask_chatgpt4(stackoverflow_questions)

    # Write the answers to a text file
    with open(output_file, "w", encoding="utf-8") as txt_file:
        for question, answer_choices in answers.items():
            txt_file.write(f"Question: {question}\n")
            for i, answer in enumerate(answer_choices, 1):
                txt_file.write(f"Answer {i}:\n{answer}\n")
            txt_file.write("\n")

    print(f"Results saved to {output_file}")
