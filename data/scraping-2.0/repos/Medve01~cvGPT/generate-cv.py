import json
import html2text
import click
import openai
import requests

config = {}

def get_config():
    """ Reads the config file and returns the config as a dict """
    # read config.json, let's use json.loads
    with open('config.json', 'r', encoding='UTF-8') as config_file:
        config_text = config_file.read()
        return json.loads(config_text)


def proofread(cv_file):
    """ Sends the CV file for proofreading to GPT and saves the output as a new file """
    if cv_file is None:
        print('No CV file provided')
        return
    print('Proofreading CV ' + cv_file)
    openai.api_key = config['openai_api_key']
    with open(cv_file, 'r', encoding='UTF-8') as cv_file:
        cv_text = cv_file.read()
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages = [
                    {"role": "system", "content": """Your task is to proofread the CV below and correct any mistakes.
                    Make sure to start the CV with a header that says [CV]."""},
                    {"role": "user", "content": cv_text}
                ]
            )
        except openai.error.InvalidRequestError as error:
            print(error)
            return
        gpt_text = response.choices[0].message.content
        # in case GPT starts babbling stuff like "Your CV looks good, but... whatever."
        proofread_cv = gpt_text.split('[CV]')[1]
        gpt_message = gpt_text.split('[CV]')[0]
        if len(gpt_message) > 0:
            print('GPT says: ' + gpt_message)
        print('GPT Proofread CV is saved under proofread-cv.txt')
        with open('proofread-cv.txt', 'w', encoding='UTF-8') as proofread_file:
            proofread_file.write(proofread_cv)

def scrape(url):
    """ Scrapes the job posting from the given URL and saves it as a new file """
    if url is None:
        print('No URL provided')
        return
    print('Scraping Job posting from url ' + url)
    h2t = html2text.HTML2Text()
    h2t.ignore_links = True
    h2t.ignore_images = True
    h2t.ignore_emphasis = True
    h2t.ignore_tables = True
    text = h2t.handle(requests.get(url).text)
    print('Job posting is saved under job-posting.txt')
    with open('job-posting.txt', 'w', encoding='UTF-8') as job_posting_file:
        job_posting_file.write(text)

def generate(cv_file):
    """ Generates a new CV based on the given CV file and the job posting using GPT """
    if cv_file is None:
        print('No CV file provided')
        return
    print('Generating CV from input ' + cv_file, 'and job posting job-posting.txt')
    openai.api_key = config['openai_api_key']
    with open(cv_file, 'r', encoding='UTF-8') as cv_file:
        cv_text = cv_file.read()
        with open('job-posting.txt', 'r', encoding='UTF-8') as job_posting_file:
            job_posting_text = job_posting_file.read()
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages = [
                        {"role": "system", "content": """
                         Your task is to fine-tune a CV based on the CV and a job posting below.
                         Make sure to understand the key requirements for the job, including years of experience, skills, position name.
                         Tailor the CV to the job posting. Start with a summary that highlights matching skills and experience.
                         Feel free to add or remove sections as needed and change text (without changing the meaning).
                         Also, feel free to cut parts that are not relevant at all.
                         Use Markdown formatting.
                         Make sure to start the CV with [CV]
                         """},
                        {"role": "user", "content": "CV: " + cv_text},
                        {"role": "user", "content": "Job Posting: " + job_posting_text}
                    ],
                )
            except openai.error.InvalidRequestError as error:
                print(error)
                return
            generated_cv = response.choices[0].message.content.split("[CV]")[1]
            print('Fine-tuned CV is saved under generated-cv.txt')
            with open('generated-cv.txt', 'w', encoding='UTF-8') as generated_file:
                generated_file.write(generated_cv)
            print('Generating cover letter')
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages = [
                        {"role": "system", "content": """
                         Your task is to generate a short cover letter based on the job posting and the CV below.
                         Make sure to understand the key requirements for the job, including years of experience, skills, position name.
                         Tailor the cover letter to the job posting. Start with a summary that highlights matching skills and experience.
                         Feel free to add or remove sections as needed and change text (without changing the meaning).
                         Also, feel free to cut parts that are not relevant at all.
                         Make sure to start the cover letter with [Cover Letter]
                         """},
                        {"role": "user", "content": "CV: " + cv_text},
                        {"role": "user", "content": "Job Posting: " + job_posting_text}
                    ],
                )
            except openai.error.InvalidRequestError as error:
                print(error)
                return
            generated_cover_letter = response.choices[0].message.content.split("[Cover Letter]")[1]
            print('Generated cover letter is saved under generated-cover-letter.txt')
            with open('generated-cover-letter.txt', 'w', encoding='UTF-8') as generated_file:
                generated_file.write(generated_cover_letter)

@click.command()
@click.option('--cv_file', '-cv', help='CV file', required=False)
@click.option('--url', '-u', help='URL of the job posting', required=False)
@click.option('--action', '-a', help='Action to perform. Valid values are: proofread, scrape, generate', required=True)
def main(cv_file, url, action):
    """ Proofreads a CV, scrapes a job posting, or generates a new CV based on the job posting. \n
    If you want to proofread a CV, use the --cv_file option, vith --action proofread \n
    If you want to scrape a job posting, use the --url option, with --action scrape\n
    If you want to generate a new CV, use the --cv_file option, with --action generate (this requires a job posting to be scraped first). A cover letter will be generated as well.\n
    the CV file must be plain text, the generated CV will be saved as md (Markdown)
    """
    match action:
        case 'proofread': proofread(cv_file)
        case 'scrape': scrape(url)
        case 'generate': generate(cv_file)
        case _: print('Invalid action')

if __name__ == '__main__':
    config = get_config()
    main() # pylint: disable=no-value-for-parameter
