import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")
COMPLETIONS_MODEL = "text-davinci-003"

def get_jd_keywords(jd_content):
    """
    Returns the keywords from a job description
    """
    prompt = """
    Extract the job title (without company name) and a list of industry relevant skills (each skill should be 1-3 words) 
    required for this job (Don't include incomplete words in your response and respond NA if anything is not present): \n
    """
    prompt += jd_content
    prompt += "\nA:"
    response = openai.Completion.create(
        engine=COMPLETIONS_MODEL,
        prompt=prompt,
        max_tokens=40,
        temperature=0.7,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        best_of=1
    )
    response_text = (response["choices"][0]["text"].strip(" \n"))
    text_split = response_text.split("\n")

    job_title = text_split[0].split(':')[1].strip()
    keywords_str = text_split[1].split(':')[1].strip()
    keywords_query = ' OR'.join(keywords_str.split(','))

    return job_title+' ('+keywords_query+')'