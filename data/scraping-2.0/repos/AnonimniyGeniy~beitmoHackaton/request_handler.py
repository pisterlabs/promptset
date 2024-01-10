import openai
import models
import time

# Set up OpenAI API
openai.api_key = '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'


def generate_recommendation(person, events):
    prompt = f"Person: {person}\nAvailable events: {events}\nGenerate event recommendations in a bullet list format. " \
             f"Leave event names unchanged.\nLeave 4 most relevant events for specific person.\n"
    print(prompt)
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=500,
        temperature=0.9,
        n=1,
        timeout=60  # Увеличение времени ожидания ответа до 60 секунд
    )
    recommended_events = response.choices[0].text

    # Send prompt to ChatGPT API for generating recommendations

    recommendations = [choice['text'].strip() for choice in response.choices]
    return recommendations


def generate_recommendations(persons, events):
    # prompt = f"Person: {person}\nAvailable events: {events}\nGenerate event recommendations in a bullet list format. Leave event names unchanged.\nLeave 4 most relevant events for specific person.\n"
    # print(prompt)
    recommendations = []
    for person in persons:
        messages = [{"role": "system",
                     "content": "Generate event recommendations for each provided person in a bullet list format. Leave event "
                                "names unchanged.\nLeave 4 most relevant events for specific person. Leave only event names list in "
                                "answer."}, {"role": "system", "content": f"Available events: {events}"},
                    {"role": "user", "content": f"Person: {person}"}]
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-16k',
            messages=messages,
            max_tokens=500,
            temperature=0.7,
            n=1,
            timeout=60  # Увеличение времени ожидания ответа до 60 секунд
        )
        recommendations += [{persons.index(person): [choice['message']['content'] for choice in response.choices]}]
        # print(recommendations[-1])
        # time.sleep(20)

    # Send prompt to ChatGPT API for generating recommendations

    return recommendations
