import time
import os
import openai

from utils.logger import get_logger

model="text-davinci-003"

def get_conversation(prompt_conv, max_tokens):
    logger = get_logger(__name__, 'utils/get_conversation.log')
    try:
        os.makedirs('output/conversations/', exist_ok=True)
        model="text-davinci-003"
        max_tokens = max_tokens
        response = openai.Completion.create(
            model=model,
            prompt=prompt_conv,
            max_tokens=max_tokens
        )
        logger.info('\nres: \n' + str(response.choices[0].text).strip())
        if prompt_conv != 'Good Bye!':
            print(str(response.choices[0].text))
        else:
            print(str(response.choices[0].text))
            return None

    except Exception as e:
        logger.debug(f'An error occurred while generating conversation: {str(e)}')

        return

    while True:
        time.sleep(0.1)
        question = input('\n\nMe: ')
        logger.info('\nqst: \n' + str(question))

        if question == 'done':
            get_conversation('Good Bye!')
        else:
            get_conversation(question)

        return response
