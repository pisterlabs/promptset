import openai
import configparser
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(current_dir, "config.ini")

# Set API key
config = configparser.ConfigParser()
config.read(config_file_path)
openai.api_key = config["OpenAI"]["api_key"]


def get_summary(result):
    # Define prompt
    prompt = (
        "Summarise this group chat that occurred on Telegram, making references to who said what "
        + result
    )

    # Call API and receive response
    generated = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    # # Extract summary text from response
    # summary = generated.choices[0].text.strip()

    # # Parse and format summary as needed
    # parsed_summary = json.loads(summary)

    # Output summary to console
    return generated["choices"][0]["text"]
