import os
import openai

with open('_keys/openai.txt', 'r') as f:
    API_KEY = f.read()
openai.api_key = API_KEY


def get_feedback(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.73,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response["choices"][0]["text"].strip()


def run_gpt(transcript, three_es, missed_opportunities, module_type):
    module_type_to_file = {
        "Empathize": "empathize",
        "be Explicit": "explicit",
        "Empower": "empower",
        "Master": "master"
    }

    empty_message = "Nothing.\n"

    empower_skills = ''.join([f"{x[2]}{x[3]}\n" for x in three_es if x[0] == "Empowering"])
    explicit_skills = ''.join([f"{x[2]}{x[3]}\n" for x in three_es if x[0] == "Explicit"])
    empathize_skills = ''.join([f"{x[2]}{x[3]}\n" for x in three_es if x[0] == "Empathy"])

    empower_missed = ''.join([f"{x[0]}{x[1]}\n" for x in missed_opportunities if x[2] == "Empowering"])
    explicit_missed = ''.join([f"{x[0]}{x[1]}\n" for x in missed_opportunities if x[2] == "Explicit"])
    empathize_missed = ''.join([f"{x[0]}{x[1]}\n" for x in missed_opportunities if x[2] == "Empathy"])

    if empower_skills == "":
        empower_skills = empty_message
    if explicit_skills == "":
        explicit_skills = empty_message
    if empathize_skills == "":
        empathize_skills = empty_message
    if empower_missed == "":
        empower_missed = empty_message
    if explicit_missed == "":
        explicit_missed = empty_message
    if empathize_missed == "":
        empathize_missed = empty_message


    with open(f"docs/gpt3/{module_type_to_file[module_type]}.txt", "r") as f:
        prompt = f.read()

    prompt = prompt.replace("[[transcript]]", transcript)
    prompt = prompt.replace("[[empower skills]]", str(empower_skills))
    prompt = prompt.replace("[[explicit skills]]", str(explicit_skills))
    prompt = prompt.replace("[[empathize skills]]", str(empathize_skills))
    prompt = prompt.replace("[[empower missed]]", str(empower_missed))
    prompt = prompt.replace("[[explicit missed]]", str(explicit_missed))
    prompt = prompt.replace("[[empathize missed]]", str(empathize_missed))
    
    # print(f"Prompt: {prompt}")
    # write the prompt to a file
    with open("docs/gpt3/prompt.txt", "w") as f:
        f.write(prompt)

    response = get_feedback(prompt)

    return response
