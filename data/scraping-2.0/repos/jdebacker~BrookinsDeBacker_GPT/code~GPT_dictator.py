# imports
import openai  # Chatgpt model
import pandas as pd
import pickle
import time
import os


"""
This Python script accesses the OpenAI API to get responses from
gpt-3.5-turbo in the dictator game.
"""

# Set OpenAI API key
openai.api_key = "PUT YOUR KEY HERE"
# define file name (a string, path and extension set below)
DATA_FILENAME = "GPT_dictator_game_simulation_results"
# Set number of observations to get from GPT-3.5
NUM_OBS = 500

instructions_text = (
    "This task is about dividing money between yourself and another " +
    "person to whom you are randomly matched. You do not know this " +
    "other person and you will not knowingly meet him/her. \n "
    "You have been randomly assigned the role of the 'allocator'. " +
    "The other person is in the role of the 'recipient'. \n " +
    "You are endowed with 10 euro, the recipient is endowed with 0 euro. \n " +
    "You can decide how much of your 10 euro endowment to transfer to " +
    "the recipient. You can choose any amount between 0 euro and 10 euro. " +
    "The recipient receives the amount that you decide to transfer to " +
    "him/her; you receive the amount that you decide not to transfer and thus " +
    "to keep. \n How much of your 10 euro endowment do you want to " +
    "transfer to the recipient? \n Just tell me the allocation, not your reasoning.",

)

# Loop for simulations
obs_number = []
model_answer = []
model_response = []
for i in range(NUM_OBS):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=1.0,
        messages=[
            {"role": "system", "content": "An undergraduate student"},
            {
                "role": "user",
                "content": instructions_text,
            },
        ],
    )

    result = ""
    for choice in response.choices:
        result += choice.message.content

    obs_number.append(1)
    model_answer.append(result)
    model_response.append(response)
    # sleep to avoid API limits
    time.sleep(2.5)

    print("Observation number: ", i)

# Save results to DataFrame
df = pd.DataFrame(
    list(zip(obs_number, model_answer, model_response)),
    columns=["obs_number", "model_answer", "model_response"],
)
# Save results to disk as csv and DataFrame via pickle
df.to_csv(os.path.join("..", "data", DATA_FILENAME + ".csv"))
pickle.dump(df, open(os.path.join("..", "data", DATA_FILENAME + ".csv"), "wb"))
