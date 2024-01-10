import os
import openai


def main():
    key = os.environ.get("CHATGPT_API_KEY")
    original_script = os.environ.get("CHATGPT_SCRIPT")
    original_script_typed = os.environ.get("CHATGPT_SCRIPT_TYPED")
    exported_script = os.environ.get("CHATGPT_EXPORTED_FILE_NAME")

    openai.api_key = key

    try:
        with open(original_script) as f:
            lines = f.read()
    except:
        lines = original_script_typed

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"Add comments to this code to make it more readable: {lines}",
            }
        ],
    )

    with open(exported_script, "w") as f:
        f.write(completion.choices[0].message.content)
