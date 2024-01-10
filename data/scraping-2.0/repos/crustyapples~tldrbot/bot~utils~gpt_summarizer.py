import openai
import os

# current_dir = os.path.dirname(os.path.abspath(__file__))
# config_file_path = os.path.join(current_dir, "config.ini")

# Set API key
# config = configparser.ConfigParser()
# config.read(config_file_path)
# openai.api_key = config["OpenAI"]["api_key"]
openai.api_key = os.environ.get("OPENAI_API_KEY")


def get_summary(result):
    # Define prompt
    prompt = (
        result
        + '''Based on the above, output the following

                        "Summary: [4-5 Sentences]

                        Sentiment: [Choose between, Positive, Negative, Neutral]"'''
    )

    # Call API and receive response
    generated = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": f"{prompt}"}]
    )

    # Output summary to console
    return generated["choices"][0]["message"]["content"]
