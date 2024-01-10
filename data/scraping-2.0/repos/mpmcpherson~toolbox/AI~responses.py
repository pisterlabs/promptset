import openai
import tiktoken
import var_dump


def TikTokenTokenize(input):
    return tiktoken.encoding_for_model("text-davinci-002").encode(input)


def generate_gpt3_response(prompt, sentiment_score):
    # Define the GPT-3 API parameters based on the sentiment score
    if sentiment_score['label'] == 'POSITIVE':
        # If the sentiment score is positive,
        # use a setting for a positive response
        gpt3_parameters = {
            "temperature": 0.7,
            "max_tokens": 150
        }
    else:
        # If the sentiment score is negative or neutral,
        # use a setting for a more general response
        gpt3_parameters = {
            "temperature": 0.8,
            "max_tokens": 200
        }

    # Generate the GPT-3 response using the given parameters
    response = openai.ChatCompletion.create(
        engine="text-davinci-002",  # Replace with the engine you want to use
        prompt=prompt,
        **gpt3_parameters
    )
    var_dump.var_dump(response)

    return response
    # ['choices'][0]['text']
