import openai

# Set the OpenAI API key
openai.api_key = "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"


def generate_hashtags_from_topic(topic):
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt="#" + topic,
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # Split the generated text into individual hashtags
    hashtags = response.text.split("\n")
    return hashtags

