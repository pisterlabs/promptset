import openai
import config

openai.api_key = config.OPENAI_API_KEY


def extract_needs_gpt3(transcripts):
    needs = []

    for transcript in transcripts:
        prompt = f"Extract user needs from the following conversation: {transcript}"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=50,
            n=1,
            stop=None,
            temperature=0.5,
        )

        need = response.choices[0].text.strip()
        needs.append(need)

    return needs

transcripts = [
    "I'm worried about affordable housing.",
    "We need better healthcare services.",
    "The public transportation system is terrible.",
    "I'm concerned about climate change.",
]

needs = extract_needs_gpt3(transcripts)
print(needs)


def generate_survey_gpt3(needs):
    survey_questions = []

    for need in needs:
        prompt = f"Create a survey question that asks the user to rate the effect of {need} on their well-being."
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=50,
            n=1,
            stop=None,
            temperature=0.5,
        )

        question = response.choices[0].text.strip()
        survey_questions.append(question)

    return survey_questions

survey_questions = generate_survey_gpt3(needs)
for question in survey_questions:
    print(question)



# This example demonstrates how to use GPT-3 to extract user needs from chat transcripts and generate survey questions. Note that using the GPT-3 API will require an API key from OpenAI, and the total number of tokens (input and output) should be within the model's limits. Adjust the model parameters, such as max_tokens, temperature, and the prompt format, as needed to optimize the performance.