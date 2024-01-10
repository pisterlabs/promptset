import openai
import os

# Load your OpenAI API key
openai.api_key = ""


def get_api_response(prompt: str) -> str | None:
    text: str | None = None

    try:
        response: dict = openai.Completion.create(
            model='text-davinci-003',
            prompt=prompt,
            temperature=0,
            max_tokens=150,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0.6,
            stop=[' Human:', ' AI:']
        )

        choices: dict = response.get('choices')[0]
        text = choices.get('text')

    except Exception as e:
        print('ERROR:', e)

    return text


def update_list(message: str, pl: list[str]):
    pl.append(message)


def create_prompt(message: str, pl: list[str]) -> str:
    p_message: str = f'\nHuman: {message}'
    update_list(p_message, pl)
    prompt: str = ''.join(pl)
    return prompt


def get_bot_response(message: str, pl: list[str]) -> str:
    prompt: str = create_prompt(message, pl)
    bot_response: str = get_api_response(prompt)

    if bot_response:
        update_list(bot_response, pl)
        pos: int = bot_response.find('\nAI: ')
        bot_response = bot_response[pos + 5:]
    else:
        bot_response = 'Something went wrong...'

    return bot_response


def main():

    prompt_list: list[str] = ['you are taking directions from a human then translating it to a robot with a paint brush who has a limited vocabulary.  the vocabulary: ,jump, turn right, turn left, walk backward , walk forward , grab,  hold, let go, start painting on the ground, stop painting on the ground, walk in a circle, IF THE USER ASKS FOR AN ITEM PUT IN INSIDE BRACKETS',
                              '\nHuman: Draw two circles that do not overlap',
                              '\nAI: start painting on the ground, walk in a circle, stop painting on the ground, walk forward, start painting on the ground, walk in a circle, stop painting on the ground']

    while True:
        user_input: str = input('You: ')
        response: str = get_bot_response(user_input, prompt_list)
        print(f'Bot: {response}')


if __name__ == '__main__':

    main()