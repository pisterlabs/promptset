import openai
import re
from app.config import DevelopmentConfig

openai.api_key = DevelopmentConfig.OPENAI_KEY


def generateChatResponse(prompt):
    messages = []
    messages.append({
        "role": "system",
        "content": "You are a quiz generator. You will generate 10 quiz questions on unless instructed otherwise. Each question is followed by the correct answer and 3 wrong answers in random order. GOOD example: 1. What is the capital of Estonia? a. Helsinki b. Rome c. Tallinn d. Riga Correct answer: a 'Correct answer' should be followed by a colon and a space and the correct letter, NEVER the full correct answer. PS! Do not only write Answer: a. It is important to write Correct answer: a'"
    })

    question = {
        "role": "user",
        "content": prompt
    }
    messages.append(question)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        answer = response['choices'][0]['message']['content'].replace('\n', '<br>')
        # Extract question and answer from the cleaned answer string
        question_text = re.findall(r"(\d+\.)\s(.*?)<br>", answer)

        correct_answer = []
        answer_options = []
        for match in re.findall(r"([a-d])\. (.*?)(?=[a-d]\.|\s*Correct answer:|$)", answer, re.IGNORECASE):
            option, content = match
            if '*' in content:
                correct_option = re.search(r"(a|b|c|d)\. ?", content)
                if correct_option:
                    correct_option = correct_option.group(1)
                    content = content.replace('*', '').strip()  # Remove * symbol within answer options
                    correct_answer.append(correct_option.lower())
            else:
                content = re.sub(r"<br>", "", content)  # Remove <br> tags within answer options
                answer_options.append((option, content.strip()))

        # Extract correct answers from the answer string
        correct_answer_lines = re.findall(r"Answer:\s*([a-d])", answer, re.IGNORECASE)
        correct_answer = [option.lower() for option in correct_answer_lines]

        return {
            'question_text': [text[1] for text in question_text],
            'correct_answer': correct_answer,
            'answer_options': answer_options,
            'answer': answer
        }

    except Exception as e:
        # print(e)
        return {
            'error': 'Something went wrong please try again'
        }


