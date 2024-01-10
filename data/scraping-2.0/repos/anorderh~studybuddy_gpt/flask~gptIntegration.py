import datetime
import openai

# Stage 3 - Use ChatGPT to generate Markdown syntax
def generate_gpt_output(job, title=None):
    start = None

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": job.prompt + job.text}
        ],
        temperature=0.7,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    job.set_gpt_output(response['choices'][0]['message']['content'] + "\n\n")