import logging
import openai
import traceback
import os

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

API_KEY = os.getenv('API_KEY')
openai.api_key = API_KEY

def get_open_ai_response(code: str, ) -> str:
    ''' Get the Open AI response that explains a piece of code.\n
        This piece of code also needs two things:
        - The code must be closed by five *****
        - A question must follow after five ***** for better response

        Parameters:
        - code: str

        Returns:
        - object
            - status: 0/1,
            - message: Successful/Unsuccessful
            - explaination: either code explaination or empty string
    '''
    logger.info('Calling the function with a piece of code...')
    logger.info(code)
    try:
        response = openai.Completion.create(
            model='code-davinci-002',
            prompt=code,
            temperature=0,
            max_tokens=64,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=['****']
        )
        if len(response['choices']) > 0:
            logging.info('Successful')
            return {
                'status': 1,
                'message': 'Successful.',
                'explaination': response['choices'][0]['text']
            }
        else:
            logging.info('Unsuccessful')
            return {
                'status': 0,
                'message': 'Unsuccessful.',
                'explaination': ''
            }
    except Exception as e:
        logger.exception(f'Uncaught exception - {traceback.format_exc()}')
        return {
                'status': 0,
                'message': 'Unsuccessful.',
                'explaination': ''
            }
