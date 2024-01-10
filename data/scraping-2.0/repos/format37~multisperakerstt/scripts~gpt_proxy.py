import openai
import logging
import os
import tiktoken

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def calculate_tokens(text, model='gpt-3.5-turbo-16k'):
    # To get the tokeniser corresponding to a specific model in the OpenAI API:
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    return len(tokens)


def gpt(query, key, system="You are a helpful assistant."):
    openai.api_key = key
    
    logger.info('forecast_tokens: '+str(calculate_tokens(query)))
    answer = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-16k',
        messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": str(query)}
        ]    
    )
    logger.info('total_tokens: '+str(answer['usage']['total_tokens']))    
    return answer['choices'][0]['message']['content']


def main():
    query = "The capital of Britain"
    answer = gpt(query, "")
    logger.info(answer)
    logger.info('Done')


if __name__ == "__main__":
    main()
