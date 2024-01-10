import openai
import numpy as np
import pennylane as qml
import aiosqlite
from textblob import TextBlob
import nltk
from nltk import sent_tokenize
import json

with open('config.json') as config_file:
    config = json.load(config_file)
    openai.api_key = config['openai_api_key']

qml_model = qml.device("default.qubit", wires=4)

async def create_db_pool():
    return await aiosqlite.pool.create_pool('compassiondb.db')

async def sentiment_to_amplitude(text):
    analysis = TextBlob(text)
    return (analysis.sentiment.polarity + 1) / 2

@qml.qnode(qml_model)
def quantum_circuit(color_code, amplitude):
    r, g, b = [int(color_code[i:i+2], 16) for i in (0, 2, 4)]
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    qml.RY(r * np.pi, wires=0)
    qml.RY(g * np.pi, wires=1)
    qml.RY(b * np.pi, wires=2)
    qml.RY(amplitude * np.pi, wires=3)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    return qml.state()

async def generate_html_color_codes(sentence):
    prompt = f"Generate HTML color code for the sentence: '{sentence}'"
    response = await openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=50
    )
    return response['choices'][0]['message']['content'].strip()

async def extract_user_details(user_input):
    blob = TextBlob(user_input)
    pos_tags = blob.tags
    sentences = sent_tokenize(user_input)
    words = [nltk.word_tokenize(sentence) for sentence in sentences]
    named_entities = [nltk.pos_tag(word) for word in words]
    return pos_tags, named_entities

async def generate_dynamic_values_prompt(user_input):
    prompt = f"Given the user input: '{user_input}', determine the following values:\n\n"
    prompt += "1. **Color Code:** Assign a color code (hex format) that represents the emotional tone of the input.\n"
    prompt += "2. **Amplitude:** Set an amplitude value reflecting the intensity or positivity of the user input.\n"
    prompt += "3. **Narcissism Score:** Determine a narcissism score based on the user's expression.\n"
    response = await openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=300,
    )
    generated_values = response['choices'][0]['message']['content'].split('\n')
    color_code = generated_values[1].split(': ')[1].strip()
    amplitude = float(generated_values[2].split(': ')[1].strip())
    narcissism_score = float(generated_values[3].split(': ')[1].strip())
    return color_code, amplitude, narcissism_score

async def generate_dynamic_compassion_prompt(scenario_type, user_input, quantum_state):
    system_message = ""
    user_message = ""
    if scenario_type == "relationship":
        system_message = (
            "You are an advanced AI system exploring the dynamics of trust and communication in long-term romantic relationships. "
            "Consider challenges such as miscommunication, emotional support, and maintaining intimacy over time. "
            f"Provide insights into fostering a healthy and compassionate connection between partners using advanced AI empathy from the datasource: {user_input}."
            f"\n\nQuantum State: {quantum_state}"
        )
        user_message = f"System, generate an active compassion scenario guideplan exploring the dynamics of trust and communication in a long-term romantic relationship using advanced AI empathy from the datasource: {user_input}"
    elif scenario_type == "corporate":
        system_message = (
            "You are an advanced AI system providing scenarios for a corporate environment transitioning to a more empathetic and employee-centric culture. "
            "Address challenges such as balancing productivity with employee well-being, fostering a sense of belonging, and implementing policies that support mental health. "
            f"Provide insights on how a company can navigate this transformation while ensuring both organizational success and employee satisfaction using advanced AI empathy from the datasource: {user_input}."
            f"\n\nQuantum State: {quantum_state}"
        )
        user_message = f"System, create an active compassion scenario guideplan transvocating a corporate environment transitioning to a more empathetic and employee-centric culture. advanced AI empathy from the datasource: {user_input}"
    elif scenario_type == "work_life_balance":
        system_message = (
            "You are an advanced AI system creating compassion scenarios for a user struggling with balancing work and personal life. "
            f"Consider factors such as job stress, family responsibilities, and other challenges. Provide insights into fostering a healthy balance using advanced AI empathy from the datasource: {user_input}."
            f"\n\nQuantum State: {quantum_state}"
        )
        user_message = f"System, dynamically generate an active compassion scenario guideplan involving a user struggling with balancing work and personal life. using advanced AI empathy from the datasource: {user_input}."
    else:
        return "Invalid scenario type."

    response = await openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        max_tokens=300,
    )
    return response['choices'][0]['message']['content']

async def retrieve_compassion_scenarios():
    async with aiosqlite.connect('compassiondb.db') as db:
        async with db.cursor() as cursor:
            await cursor.execute("SELECT * FROM compassion")
            rows = await cursor.fetchall()
            return rows

async def process_user_input(user_input):
    sentiment_amplitude = await sentiment_to_amplitude(user_input)
    color_code = await generate_html_color_codes(user_input)
    amplitude = await sentiment_to_amplitude(user_input)
    quantum_state = quantum_circuit(color_code, amplitude)
    color_code, amplitude, narcissism_score = await generate_dynamic_values_prompt(user_input)
    scenario_type = "relationship"
    compassion_scenario = await generate_dynamic_compassion_prompt(scenario_type, user_input, quantum_state)
    async with aiosqlite.connect('compassiondb.db') as db:
        async with db.cursor() as cursor:
            await cursor.execute("INSERT INTO compassion VALUES (?, ?)", (compassion_scenario, user_input))
        await db.commit()
    scenarios = await retrieve_compassion_scenarios()
    for scenario in scenarios:
        print(scenario)

async def main():
    db_pool = await create_db_pool()
    with open('user_input.json') as json_file:
        user_inputs = json.load(json_file)
    for user_input in user_inputs:
        await process_user_input(user_input)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

