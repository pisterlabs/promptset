import openai

openai.api_key = 'OPENAI_API_KEY'

def generate_about_me_summary(parsed_resume_data):
    prompt = "Write a professional and personalized 'About Me' summary based on the provided resume data:\n\n" + parsed_resume_data + "\n\nSummary:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
        n=1,
        stop=None,
        temperature=0.7
    )
    return response.choices[0].text.strip()
