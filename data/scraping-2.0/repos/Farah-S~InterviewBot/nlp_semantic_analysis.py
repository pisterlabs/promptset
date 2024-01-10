

import pandas as pd
import openai
import spacy

# Set up your OpenAI API key

# Load the spaCy language model for semantic analysis
nlp = spacy.load("en_core_web_sm")

# Load the Excel file
excel_file = "Data_analyst_question.xlsx"  # Replace with your actual file path
df = pd.read_excel(excel_file)



# Ask the user to choose a category from column A (Category)
print("Available Categories:")
categories = df["Category"].unique()
for idx, category in enumerate(categories):
    print(f"{idx + 1}. {category}")

category_choice = int(input("Choose a category (enter the corresponding number): ")) - 1
selected_category = categories[category_choice]

# Filter questions and correct answers for the selected category
category_data = df[df["Category"] == selected_category]
category_data = category_data.sample(frac=1).reset_index(drop=True)  # Shuffle within the category

category_data

# Initialize a list to store user answers
user_answers = []
questions = []
correct_answers = []

# Generate questions and collect user answers
num_questions = min(2, len(category_data))  # Ensure you don't exceed the available questions
feedback = ""

for i in range(num_questions):
    question = category_data.loc[i, "Question"]
    questions.append(question)

    correct_answer = category_data.loc[i, "Correct Answer"]
    correct_answers.append(correct_answer)

    user_answer = input(f"Q: {question}\nA: ")
    user_answers.append(user_answer)

# Analyze user answers using semantic analysis and calculate similarity scores
#user_ans_doc = nlp(user_answer)
#correct_ans_doc = nlp(correct_answer)
#similarity_score = user_ans_doc.similarity(correct_ans_doc)
scores = []
for u, q, c in zip(user_answers, questions, correct_answers):
  user_ans_doc = nlp(u)
  correct_ans_doc = nlp(c)
  similarity_score = user_ans_doc.similarity(correct_ans_doc)
  scores.append(similarity_score)

# Provide personalized feedback to the user based on semantic analysis
#feedback += f"Question: {question}\n"
#feedback += f"Your Answer: {user_answer}\n"
#feedback += f"Correct Answer: {correct_answer}\n"
#feedback += f"Your Score: {similarity_score:.2f}\n"
#feedback += "\n"
for u, q, c, s in zip(user_answers, questions, correct_answers, scores):
  feedback += f"Question: {q}\n"
  feedback += f"Your Answer: {u}\n"
  feedback += f"Correct Answer: {c}\n"
  feedback += f"Your Score: {s:.2f}\n"
  feedback += "\n"
# Display personalized feedback to the user
print("\nFeedback:")
print(feedback)



# Set up your OpenAI API key
openai.api_key = "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"

# Load the spaCy language model for semantic analysis
nlp = spacy.load("en_core_web_sm")

def load_excel_file(file_path):
  """
  Load the Excel file and return the DataFrame.
  """
  df = pd.read_excel(file_path)
  return df

def choose_category(df):
  """
  Ask the user to choose a category from column A (Category).
  Return the selected category.
  """
  print("Available Categories:")
  categories = df["Category"].unique()
  for idx, category in enumerate(categories):
    print(f"{idx + 1}. {category}")

  category_choice = int(input("Choose a category (enter the corresponding number): ")) - 1
  selected_category = categories[category_choice]
  return selected_category

def filter_questions(df, selected_category):
  """
  Filter questions and correct answers for the selected category.
  Shuffle the filtered data within the category.
  Return the filtered DataFrame.
  """
  category_data = df[df["Category"] == selected_category]
  category_data = category_data.sample(frac=1).reset_index(drop=True)  # Shuffle within the category
  return category_data

def generate_questions(category_data, num_questions):
  """
  Generate questions from the category data and collect user answers.
  Return the lists of questions and correct answers.
  """
  questions = []
  correct_answers = []
  user_answers = []

  for i in range(num_questions):
    question = category_data.loc[i, "Question"]
    questions.append(question)

    correct_answer = category_data.loc[i, "Correct Answer"]
    correct_answers.append(correct_answer)

    user_answer = input(f"Q: {question}\nA: ")
    user_answers.append(user_answer)

  return questions, correct_answers, user_answers

def analyze_similarity(user_answers, questions, correct_answers):
  """
  Analyze user answers using semantic analysis and calculate similarity scores.
  Return the list of similarity scores.
  """
  scores = []
  for u, q, c in zip(user_answers, questions, correct_answers):
    user_ans_doc = nlp(u)
    correct_ans_doc = nlp(c)
    similarity_score = user_ans_doc.similarity(correct_ans_doc)
    scores.append(similarity_score)
  return scores

def provide_feedback(user_answers, questions, correct_answers, scores):
  """
  Provide personalized feedback to the user based on semantic analysis.
  """
  feedback = ""
  for u, q, c, s in zip(user_answers, questions, correct_answers, scores):
    feedback += f"Question: {q}\n"
    feedback += f"Your Answer: {u}\n"
    feedback += f"Correct Answer: {c}\n"
    feedback += f"Your Score: {s:.2f}\n"
    feedback += "\n"
  return feedback

def main():
  # Load the Excel file
  excel_file = "Data_analyst_question.xlsx"  # Replace with your actual file path
  df = load_excel_file(excel_file)

  # Ask the user to choose a category
  selected_category = choose_category(df)

  # Filter questions for the selected category
  category_data = filter_questions(df, selected_category)

  # Generate questions and collect user answers
  num_questions = min(2, len(category_data))  # Ensure you don't exceed the available questions
  questions, correct_answers, user_answers = generate_questions(category_data, num_questions)

  # Analyze user answers using semantic analysis
  scores = analyze_similarity(user_answers, questions, correct_answers)

  # Provide personalized feedback to the user
  feedback = provide_feedback(user_answers, questions, correct_answers, scores)

  # Display personalized feedback to the user
  print("\nFeedback:")
  print(feedback)

if __name__ == "__main__":
  main()
