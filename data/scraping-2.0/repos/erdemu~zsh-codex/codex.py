import openai
import keyring
import sys
import os


def get_api_key():
    # check api key exists in keyring
    key = keyring.get_password("zsh_codex_open_ai_api_key", "key") 
    if key is None:
        # if not error
        print("No API key found in keyring")
        sys.exit(1)
    else:
        return key

def main():
    # get all the words passed in as arguments
    words = sys.argv[1:]
    # combinde them with a space
    phrase = ' '.join(words)
    org_phrase = phrase
    phrase = phrase.strip()
    phrase = "# zsh terminal " + phrase
    # append a new line
    phrase += '\n'

    # get the api key
    openai.api_key = get_api_key() 

    # Try to get env variable for engine selection
    engine = os.getenv('ZSH_CODEX_OPENAI_ENGINE')

    if engine is None:
        engine = "code-davinci-002"

    # call the openai api
    response = openai.Completion.create(
        engine=engine,
        prompt=phrase,
        temperature=0.0,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"]
    )
    text = response['choices'][0].text
    # remove leading + if it exists
    text = text.strip()
    text = text.lstrip('+')
    print(f"{text}")

if __name__ == "__main__":
    main()
