import json
import os
import wandb
import openai
import getpass


# OpenAI API key
if os.getenv("OPENAI_API_KEY") is None:
    if any(['VSCODE' in x for x in os.environ.keys()]):
        print('Please enter password in the VS Code prompt at the top of your VS Code window!')
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Paste your OpenAI Key from: https://platform.openai.com/account/api-keys\n")
    openai.api_key = os.getenv("OPENAI_API_KEY", "")

assert os.getenv("OPENAI_API_KEY", "").startswith("sk-"), "This doesn't look like a valid OpenAI API key"
print("OpenAI API key configured")

# Set up WandB project and run
wandb.init(project="evaluation-dataset")

# locations and destinations for evaluation
evaluation_data = [
    {"location": "Ikeja", "destination": "National Stadium"},
    {"location": "Lekki", "destination": "Ojo"},
    {"location": "Masha", "destination": "Doyin"},
    {"location": "Surulere", "destination": "Badagry"},
    {"location": "Ikeja", "destination": "National Stadium"},
    {"location": "Lekki", "destination": "Ojo"},
    {"location": "Masha", "destination": "Doyin"},
    {"location": "Berger", "destination": "Festac"},
    {"location": "Computer Village", "destination": "Maryland"},
    {"location": "Iponri", "destination": "Aguda"},
    {"location": "Mile-2", "destination": "Iyana-Ipaja"},
    {"location": "Iyana Ipaja", "destination": "Costain"},
    {"location": "Epe", "destination": "Ebute-Meta"},
    {"location": "Costain", "destination": "LUTH"},
    {"location": "Idi-Araba", "destination": "Doyin"},
    {"location": "Admiralty Road, Lekki Phase 1", "destination": "Lekki, Phase 2"},
    {"location": "Ikeja", "destination": "Yaba"},
    {"location": "Ojo Barracks", "destination": "Trade-Fair"}
]

# An empty list to store evaluation examples
evaluation_examples = []

# Generate conversations for evaluation
for data in evaluation_data:
    location = data["location"]
    destination = data["destination"]

    user_messages = [
        {"role": "user", "content": f"Location: {location}."},
        {"role": "user", "content": f"Destination: {destination}."}
    ]

    # Modify the expected assistant reply for each location-destination pair
    expected_reply = ""

    if location == "Ikeja" and destination == "National Stadium":
        expected_reply = "Na straight na. Take Mobolaji Bank Anthony Way, then turn right onto Ikorodu Road. Follow Ikorodu Road all the way to the stadium."
    elif location == "Lekki" and destination == "Ojo":
        expected_reply = "Take Lekki-Epe Expressway to Ozumba Mbadiwe Avenue. Then turn left onto the Third Mainland Bridge. Follow the Third Mainland Bridge all the way to Ojo."
    elif location == "Masha" and destination == "Doyin":
        expected_reply = "Take Apapa-Oshodi Expressway to Costain. Then turn left onto Ijora Causeway. Follow Ijora Causeway to Iganmu. Then turn right onto Doyin Street."
    elif location == "Surulere" and destination == "Badagry":
        expected_reply = "Take Oshodi-Apapa Expressway to Badagry Expressway. Follow Badagry Expressway all the way to Badagry."
    elif location == "Berger" and destination == "Festac":
        expected_reply = "Take Lagos-Abeokuta Expressway to Festac Town Exit. Then turn left onto Festac Link Road. Follow Festac Link Road to Festac Town."
    elif location == "Computer Village" and destination == "Maryland":
        expected_reply = "Take Ikorodu Road to Ikeja Under Bridge. Then turn left onto Mobolaji Bank Anthony Way. Follow Mobolaji Bank Anthony Way to Maryland."
    elif location == "Iponri" and destination == "Aguda":
        expected_reply = "Take Lagos Island Ring Road to Fatai Atere Way. Then turn right onto Igbosere Road. Follow Igbosere Road to Aguda."
    elif location == "Mile-2" and destination == "Iyana-Ipaja":
        expected_reply = "Take Lagos-Badagry Expressway to Iyana-Ipaja Exit. Then turn right onto Iyana-Ipaja Road. Follow Iyana-Ipaja Road to Iyana-Ipaja."
    elif location == "Iyana Ipaja" and destination == "Costain":
        expected_reply = "Take Iyana-Ipaja Road to Lagos-Abeokuta Expressway. Then turn left onto Oshodi-Apapa Expressway. Follow Oshodi-Apapa Expressway all the way to Costain."

    evaluation_example = {
        "messages": user_messages,
        "expected_reply": expected_reply
    }

    evaluation_examples.append(evaluation_example)

# Log evaluation examples to WandB
wandb.log({"evaluation_examples": evaluation_examples})

# Save the evaluation dataset to a JSON file
with open("evaluation_dataset.jsonl", "w") as jsonl_file:
    for example in evaluation_examples:
        json.dump(example, jsonl_file)
        jsonl_file.write("\n")

# Log the evaluation dataset to WandB
artifact = wandb.Artifact(name="evaluation_dataset", type="dataset")
artifact.add_file("evaluation_dataset.jsonl")
wandb.run.log_artifact(artifact)

print(f"Evaluation dataset saved to 'evaluation_dataset.jsonl' with {len(evaluation_examples)} examples.")
