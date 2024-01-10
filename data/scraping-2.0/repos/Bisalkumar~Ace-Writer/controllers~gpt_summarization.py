import openai


# enter your OpenAI API key here
openai.api_key = 'sk-#r7M0T1Qs0QrkMvhM0gFnT3BlbkFJfYOl5UILfBNLHtMOmPs8'


def openai_summarizer(text):
    prompt = f'Summarize the following text: {text}'
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5
    )

    summary = response.choices[0].text.strip()
    return summary
