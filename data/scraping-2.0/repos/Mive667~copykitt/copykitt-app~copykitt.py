import os
import argparse
from openai import OpenAI

MAX_INPUT_LENGTH = 20
MAX_EKYWORDS_LENGTH = 5
RESPONSE_LENGTH = 35
MAX_RESPONSE_LENGTH = 50

# Get the API key
# Initialize the OpenAI client with the API key
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# define an entry point for the app
def main():
    """ send 2 requests to get branding snippet and keywords respectively """

    print('Running Copy Kkit')
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    args = parser.parse_args()
    user_input = args.input
    # command: python3 copykkit.py -i "coffee mug"
    print(f'your input is: {user_input}')

    # limit the input length
    if validate_propt(user_input):
        answer, keywords_answer = generate_branding_snippet(user_input)
        # keyword_answer = generate_keywords(user_input)
        print(f'answer: {answer}')
        print(f'keyword answer: {keywords_answer}')
    else:
        raise ValueError(
            f'Prompt is too long! Must not exceed {MAX_INPUT_LENGTH} characters. your imput is {user_input}.'
        )


def generate_branding_snippet(prompt: str) -> (str, str):
    """ Generate branding snippet and keywords based on user input

    Args:
        prompt str: product name

    Returns:
        str: bradning snippet
        list[str]: keywords
    """
    # limitation of the number of response tokens
    enriched_prompt = f'Generate a upbeat branding snippet for {prompt},  the length of snippet should not exceed {RESPONSE_LENGTH} words. your snippet must be: complete.'

    # get branding snippet from api call
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": enriched_prompt},
        ],
        max_tokens=MAX_RESPONSE_LENGTH,
    )
    branding_snippet = completion.choices[0].message.content

    # get keywords from api call, based on conversation history
    keyword_prompt = f'this is a branding snippet of {prompt}, please extract no more than {MAX_EKYWORDS_LENGTH} keywords from this snippet, and return them in a square bracket. \n branding_snippet:{branding_snippet}'
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": keyword_prompt}
        ],
        max_tokens=MAX_RESPONSE_LENGTH,
    )
    keywords = completion.choices[0].message.content
    # keywords_list = re.split(r"\[|\]|, ", keywords)
    keywords_list = keywords.replace('[', '').replace(']', '').split(', ')

    return branding_snippet.strip('"').replace('\\', ''), keywords_list


def validate_propt(prompt: str) -> bool:
    return len(prompt) <= MAX_INPUT_LENGTH


if __name__ == '__main__':
    main()
