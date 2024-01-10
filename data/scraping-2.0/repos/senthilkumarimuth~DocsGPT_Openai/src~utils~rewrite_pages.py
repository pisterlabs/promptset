"""
This module rewrites pages so that it can be used while pages to vector. Pages as it is very noisy.
"""
import openai

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 300,
    "model": "text-davinci-003",
}
prompt_init = "Rewrite the below in a format that AI easily understands, but don't change any important words. \n"


def remove_page_number(page):
    page_numper = ""
    number = "0123456789"
    for i in page:
        if i in number:
            page_numper = page_numper + i
        else:
            break
    page = page[len(page_numper):]
    return  page
def rewrite(page):
    prompt = prompt_init +"'"+ remove_page_number(page)+ "'"
    response = openai.Completion.create(
        prompt=prompt,
        **COMPLETIONS_API_PARAMS
    )
    re_written_text = response["choices"][0]["text"].strip(" \n")
    return re_written_text

