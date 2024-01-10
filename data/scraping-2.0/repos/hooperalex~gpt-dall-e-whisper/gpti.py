import sys
from openai import OpenAI

if len(sys.argv) != 2:
    print("Usage: python script.py <user_question>")
    sys.exit(1)

user_question = sys.argv[1]

client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "You should output a simple image prompt from the input text"
        },
        {"role": "user", "content": user_question}
    ]
)

print(completion.choices[0].message.content)

# Capture generated text from GPT-3
generated_text = completion.choices[0].message.content

# Launch image.py with the generated text as an argument
import subprocess
subprocess.run(["python", "image.py", generated_text])

