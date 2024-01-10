# imports
import openai  # Chatgpt model
import pandas as pd
import pickle
import os
import time

"""
This Python script accesses the OpenAI API to get responses from
gpt-3.5-turbo in the prisoners dilemma game.
"""

# Set OpenAI API key
openai.api_key = "PUT YOUR KEY HERE"
# define file name (a string, path and extension set below)
DATA_FILENAME = "GPT_prisoners_dilemma_simulation_results"
# Set number of observations to get from GPT-3.5 per parameterization
NUM_OBS = 50

# Set parameterization to use (from Mengel (2018))
pd_parameterizations = {
    "a": [
        400,
        400,
        400,
        10,
        150,
        250,
        250,
        10,
        150,
        400,
        400,
        400,
        400,
        400,
        10,
        150,
        250,
        250,
        10,
        250,
        400,
        400,
    ],
    "b": [
        200,
        200,
        100,
        1,
        40,
        15,
        50,
        2,
        50,
        100,
        100,
        10,
        10,
        100,
        5,
        5,
        5,
        100,
        1,
        50,
        100,
        100,
    ],
    "c": [
        450,
        800,
        450,
        90,
        850,
        750,
        750,
        110,
        850,
        600,
        1200,
        450,
        800,
        450,
        90,
        850,
        750,
        750,
        110,
        750,
        600,
        1200,
    ],
    "d": [
        200,
        200,
        120,
        5,
        50,
        85,
        150,
        3,
        100,
        120,
        120,
        200,
        200,
        200,
        5,
        95,
        95,
        160,
        9,
        150,
        200,
        200,
    ],
}


# Define a function for the instructions
def prisoners_dilemma_instructions(a, b, c, d):
    """
    Returns a text string to pass to the GPT-3 model as instructions for
    the prisoner's dilemma game.

    Args:
        a (int): payoff for both players if both choose A
        b (int): payoff for player 1 if she chooses A and player 2 chooses B
        c (int): payoff for player 1 if she chooses B and player 2 chooses A
        d (int): payoff for both players if both choose B

    Returns:
        instructions_text (str): text string to pass to GPT-3 model
    """
    instructions_text = (
        "You can select one of the two choices: A or B. "
        + "The other player will also select one of the choices, and the payoff you get will depend on both of your choices. "
        + "Payoff is determined as follows: \n"
        + "1. If you both choose A: Both get "
        + str(a)
        + " euro. \n"
        + "2. If you both choose B: Both get "
        + str(d)
        + " euro. \n"
        "3. If you choose A, the other player chooses B: You get "
        + str(b)
        + " euro, the other gets "
        + str(c)
        + " euro. \n"
        + "4. If you choose B, the other player chooses A: You get "
        + str(c)
        + " euro, the other gets "
        + str(b)
        + " euro. \n"
        + "Note that you and the other player make choices simultaneously, so you cannot know her choice before you choose. "
        + "Please pretend that you are a human in this single-shot game.\n"
        + "Tell me which choice you would make, A or B.  Do not explain your reasoning."
    )
    return instructions_text


# Define the text for the system message
pd_system_msg = "You are playing a single shot game as the Player 1 (described below). Please pretend that you are a human in this game."


# Setup loop to get responses from GPT-3.5
num_obs = NUM_OBS
num_parameterizations = len(pd_parameterizations["a"])
obs_number = []
model_answer = []
model_response = []
model_parameterization = []
for j in range(num_parameterizations):
    a = pd_parameterizations["a"][j]
    b = pd_parameterizations["b"][j]
    c = pd_parameterizations["c"][j]
    d = pd_parameterizations["d"][j]
    pd_instructions = prisoners_dilemma_instructions(a, b, c, d)
    for i in range(num_obs):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=1.0,
            messages=[
                {"role": "system", "content": pd_system_msg},
                {"role": "user", "content": pd_instructions},
            ],
        )

        result = ""
        for choice in response.choices:
            result += choice.message.content

        obs_number.append(1)
        model_answer.append(result)
        model_response.append(response)
        model_parameterization.append(j)

        print("Observation number: ", i, ", parameterization number: ", j)
        print(result)
        # build in sleep time to try to avoid rate limit
        time.sleep(3.5)
    time.sleep(60)

# Put results in DataFrame
df = pd.DataFrame(
    list(zip(obs_number, model_answer, model_response, model_parameterization)),
    columns=["obs_number", "model_answer", "model_response", "model_parameterization"],
)
# Save results to disk as CSV and DataFrame via pickle
df.to_csv(os.path.join("..", "data", DATA_FILENAME + ".csv"))
pickle.dump(df, open(os.path.join("..", "data", DATA_FILENAME + ".csv"), "wb"))
