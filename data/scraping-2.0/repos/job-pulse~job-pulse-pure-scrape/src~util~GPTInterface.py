import openai
import logging
from models import JobCategory, ExperienceCategory
from dotenv import load_dotenv
import os
import time

load_dotenv()  # Load environment variables from .env file
openai.api_key = os.getenv("OPENAI_API_KEY")

try:
    response = openai.Completion.create(engine="davinci", prompt="Testing for key validity", max_tokens=50)
    logging.info(response.choices[0].text.strip())
    openai_key = True

except Exception as e:
    logging.info(f"Error: {e}")
    openai_key = False

NUMBER_PRINTER = "You are a helpful number printer that prints a number and the number only."
YES_NO_PRINTER = "You are a helpful yes/no printer that prints 'yes' or 'no' and the word only."

MAX_RETRIES = 2
RETRY_DELAY = 5  # in seconds

def ask_gpt(input_ask, role):
    attempts = 0
    if not openai_key:
        return ''
    while attempts < MAX_RETRIES:
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": role},
                    {"role": "user", "content": input_ask}
                ]
            )
            res = completion.choices[0].message.content
            if isinstance(res, str):
                return res.lower().strip().strip('.')
        except Exception as e:
            logging.error(f"Error in ask_gpt: {e}. Retrying...")
            time.sleep(RETRY_DELAY)
            attempts += 1
    return None

def gpt_get_yoe_from_jd(JDText):
    question = "What's the minimum number of work experience for this job? Here is the job description\n" + '"""' + JDText + '"""\n' + "You MUST say a number and SAY THE NUMBER ONLY. If there is no number mentioned in the text, print '0' and '0' only.\n The number is "
    yoe_requirement = ask_gpt(question, NUMBER_PRINTER)
    try:
        return int(yoe_requirement)
    except:
        return -1

def gpt_check_job_title(job_title, job_category, experience_category=None, yoe=None):
    intern_extra_keyword = ""
    ng_extra_keyword = ""
    if experience_category == ExperienceCategory.Intern:
        intern_extra_keyword = "internship "
    elif yoe and isinstance(yoe, int) and yoe < 2:
        ng_extra_keyword = "entry level "

    if job_category == JobCategory.Software_Engineer:
        question = f"Given the job title '{job_title}', does it look like a {ng_extra_keyword}software engineer related or IT related {intern_extra_keyword}role? Answer 'yes' or 'no' with no punctuation."
    elif job_category == JobCategory.Product_Manager:
        question = f"Given the job title '{job_title}', does it look like a product manager related role? Answer 'yes' or 'no' with no punctuation."

    gpt_answer = ask_gpt(question, YES_NO_PRINTER)
    if not gpt_answer:
        logging.error("Error in gpt_check_job_title: No answer received from GPT. Returning True to be safe.")
        return True
    return "yes" in gpt_answer
