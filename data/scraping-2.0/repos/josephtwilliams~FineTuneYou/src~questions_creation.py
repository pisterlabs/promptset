import os
import openai
import logging
from dotenv import load_dotenv

# Custom logging format
logging.basicConfig(format='%(message)s', level=logging.INFO)

def generate_resume_questions(api_key=None, model_engine="gpt-3.5-turbo", temperature=0.2, max_tokens=3000, resume_file="./src/text_files/resume.txt", output_file="./src/text_files/resume_questions.txt"):
    # Load environment variables if API key is not provided
    if api_key is None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
    
    if api_key is None:
        logging.error("No OpenAI API Key found.")
        return

    openai.api_key = api_key

    try:
        # Read resume text
        with open(resume_file, "r") as f:
            resume_text = f.read()

        # Prepare the prompt
        prompt = (
            "Based on the resume provided below, please generate a list of interview questions and their corresponding answers in the following format:\n"
            "Question: [Your question here]\n"
            "Answer: [Answer based on the resume here]\n\n"
            "For example:\n"
            'Question: Where did you go to university?\n'
            'Answer: I went to The University of Michigan.\n'
            f"\n---\n\nActual Resume:\n{resume_text}\n\n"
        )

        # Generate completion
        completion = openai.ChatCompletion.create(
            model=model_engine,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Save output
        qa_output = completion['choices'][0]['message']['content'].strip()
        with open(output_file, "w") as f:
            f.write(qa_output)
        
        logging.info("Successfully generated questions and answers from resume text file")
    except Exception as e:
        logging.error(f"Error in generating questions: {e}")

if __name__ == "__main__":
    generate_resume_questions()
