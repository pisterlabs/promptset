import openai
from info import *
from web_driver import *


def solve() -> None:
    """
    Use OpenAI to generate a thesis sentence for a Latin composition assignment.

    This function uses OpenAI to generate a thesis sentence for a Latin composition assignment based on the content of a
    web page. It retrieves a Latin poem, a prompt, and specific requirements from the web page and then constructs a
    question to provide to OpenAI's model. After receiving a response from OpenAI, it enters the generated thesis
    sentence into the web page.

    :return: None
    """
    
    global openai_enabled
    global openai_token
    global openai_model
    
    if openai_enabled == False or openai_token == "none":
        return None

    openai.api_key = openai_token

    catullus_poem = str(driver.find_element(By.ID, 'translation_write').text)
    catullus_prompt = str(driver.find_element(By.XPATH, "// div[@class='ui-body ui-body-a']"))
    thesis_input = driver.find_element(By.ID, 'thesis_write')

    requirements = ['must be one sentence and kept short', 'use multiple verbs', 'include the name of the poem in the sentence']
    question = f'Write a thesis using this poem: {catullus_poem}\n\n The following prompt: {catullus_prompt}\n\n and follow these requirements:'

    for requirement in requirements:
        question += f'{requirement}\n'

    response = openai.ChatCompletion.create(model=openai_model, messages=[{"role": "user", "content": question}])

    thesis_input.clear()
    thesis_input.send_keys(str(response.choices[0].message.content))
    thesis_input.send_keys(Keys.ENTER)