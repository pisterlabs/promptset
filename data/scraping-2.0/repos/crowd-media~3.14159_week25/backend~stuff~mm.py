import yaml
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def generate_chat(prompt_filename: str):
    with open(prompt_filename, "r") as config_file:

        try:
            prompt = yaml.safe_load(config_file)
        except yaml.YAMLError as exc:
            print(exc)

    # mem_template = "\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?\nHuman:"

    system_initial_message = {"role": "system", "content": f"{prompt['prompt']}"}

    system_simulation_user_initial_message = {
        "role": "system",
        "content": f"{prompt['user_prompt']}",
    }

    messages = []
    messages.append(system_initial_message)

    simulation_user_messages = []
    simulation_user_messages.append(system_simulation_user_initial_message)

    total_turns = 4

    simulation_user_question = "hi"

    # print(prompt)
    for turn in range(total_turns):

        ## assistant turn

        question = simulation_user_question

        new_user_message = {"role": "user", "content": f"{question}"}

        messages.append(new_user_message)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,  # randomness degree papa,
            max_tokens=2048,  # limit output
            messages=messages,
        )

        # print(response.usage.total_tokens)
        # exit()
        total_tokens = response.usage.total_tokens
        print(f"total tokens: {total_tokens}")
        answer = response.choices[0].message.content
        assistant_message = response.choices[0].message
        messages.append(assistant_message)
        print(answer)

        ## fake user simulation

        simulation_question = assistant_message

        simulation_user_message = {"role": "user", "content": f"{simulation_question}"}
        simulation_user_messages.append(simulation_user_message)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,  # randomness degree papa,
            max_tokens=2048,  # limit output
            messages=simulation_user_messages,
        )

        total_tokens = response.usage.total_tokens
        print(f"total tokens: {total_tokens}")
        sim_answer = response.choices[0].message.content
        sim_message = response.choices[0].message
        simulation_user_messages.append(sim_message)

        ## cp sim_answer for next turn
        simulation_user_question = sim_answer
        print(sim_answer)

        ### refree call

    return


generate_chat("prompt.yaml")
