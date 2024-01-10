import os
import openai


def main():
    key = os.environ.get("api_key")
    meeting_notes = os.environ.get("CHATGPT_TEXT_FILE")
    exported_file_name = os.environ.get("CHATGPT_DESTINATION_FILE_NAME")

    openai.api_key = key

    with open(meeting_notes) as f:
        lines = f.readlines()
    text = lines[0]

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"Create a quick summary of my meeting notes and also include specific next steps for each person in a separate list of bullet points. Correct any grammatical mistakes and call out if there are any unanswered questions. Here are my meeting notes: {text}",
            }
        ],
    )

    with open(exported_file_name, "w") as f:
        f.write(completion.choices[0].message.content)
