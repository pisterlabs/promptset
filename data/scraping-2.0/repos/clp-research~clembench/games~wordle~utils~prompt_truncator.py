import tiktoken
import openai

import backends

NAME = "openai"

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    supported_models = [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0301",
        "gpt-4",
        "gpt-4-0314",
        "text-davinci-003",
    ]
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in supported_models:  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += (
                4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            )
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not presently implemented for model {model}."""
        )


def test_counting():
    messages = [
        {
            "role": "system",
            "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English.",
        },
        {
            "role": "system",
            "name": "example_user",
            "content": "New synergies will help drive top-line growth.",
        },
        {
            "role": "system",
            "name": "example_assistant",
            "content": "Things working well together will increase revenue.",
        },
        {
            "role": "system",
            "name": "example_user",
            "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage.",
        },
        {
            "role": "system",
            "name": "example_assistant",
            "content": "Let's talk later when we're less busy about how to do better.",
        },
        {
            "role": "user",
            "content": "This late pivot means we don't have time to boil the ocean for the client deliverable.",
        },
    ]

    for model in ["gpt-3.5-turbo-0301", "gpt-4", "gpt-4-0314"]:
        print(model)
        # example token count from the function defined above
        print(
            f"{num_tokens_from_messages(messages, model)} prompt tokens counted by num_tokens_from_messages()."
        )
        # example token count from the OpenAI API
        creds = backends.load_credentials(NAME)
        if "organisation" in creds[NAME]:
            openai.organization = creds[NAME]["organisation"]
        openai.api_key = creds[NAME]["api_key"]        

        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=1,  # we're only counting input tokens here, so let's not waste tokens on the output
        )
        print(
            f'{response["usage"]["prompt_tokens"]} prompt tokens counted by the OpenAI API.'
        )
        print()


def test_counting_wordle_response():
    guess_messages = [
        {
            "guess:": "otter",
            "explanation:": "This is a five-letter English word that is a relative of the mink.",
        }
    ]
    agreement_messages = [
        {
            "agreement:": "yes",
            "explanation:": "This is a five-letter English word that is a relative of the mink.",
        }
    ]

    print(
        f'Guesser Response has {num_tokens_from_messages(guess_messages, "gpt-3.5-turbo")} tokens'
    )
    print(
        f'Critic Response has {num_tokens_from_messages(agreement_messages, "gpt-3.5-turbo")} tokens'
    )


if __name__ == "__main__":
    #test_counting_wordle_response()
    test_counting()
