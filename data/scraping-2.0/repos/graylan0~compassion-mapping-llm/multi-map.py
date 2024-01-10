import openai
import numpy as np
import pennylane as qml
import aiosqlite
from textblob import TextBlob
import nltk
from nltk import sent_tokenize

openai.api_key = 'YOUR_OPENAI_API_KEY'

qml_model = qml.device("default.qubit", wires=4)

def sentiment_to_amplitude(text):
   try:
       analysis = TextBlob(text)
       return (analysis.sentiment.polarity + 1) / 2
   except Exception as e:
       print(f"An error occurred: {e}")

@qml.qnode(qml_model)
def quantum_circuit(color_code, amplitude):
   try:
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
   except Exception as e:
       print(f"An error occurred: {e}")

async def generate_html_color_codes(sentence):
   try:
       prompt = f"Generate HTML color code for the sentence: '{sentence}'"
       response = openai.ChatCompletion.create(
           model="gpt-3.5-turbo",
           messages=[
               {"role": "system", "content": "You are a helpful assistant."},
               {"role": "user", "content": prompt},
           ],
           temperature=0.7,
           max_tokens=50
       )
       color_code = response['choices'][0]['message']['content'].strip()
       return color_code
   except Exception as e:
       print(f"An error occurred: {e}")

def extract_user_details(user_input):
   try:
       blob = TextBlob(user_input)
       pos_tags = blob.tags

       sentences = sent_tokenize(user_input)
       words = [nltk.word_tokenize(sentence) for sentence in sentences]
       named_entities = [nltk.pos_tag(word) for word in words]

       return pos_tags, named_entities
   except Exception as e:
       print(f"An error occurred: {e}")
async def generate_dynamic_compassion_prompt(scenario_type, user_input=None):
 try:
     if scenario_type == "relationship":
         system_message = (
             "You are an advanced AI system exploring the dynamics of trust and communication in long-term romantic relationships. "
             "Consider challenges such as miscommunication, emotional support, and maintaining intimacy over time. "
             "Provide insights into fostering a healthy and compassionate connection between partners."
         )
         user_message = "System, generate a dialogue possible between relationship parties using compassion scenario inspection exploring the dynamics of trust and communication in a long-term romantic relationship."

     elif scenario_type == "corporate":
         system_message = (
             "You are an advanced AI system providing scenarios for a corporate environment transitioning to a more empathetic and employee-centric culture. "
             "Address challenges such as balancing productivity with employee well-being, fostering a sense of belonging, and implementing policies that support mental health. "
             "Provide insights on how a company can navigate this transformation while ensuring both organizational success and employee satisfaction."
         )
         user_message = "System, create a compassion scenario inspection for commiting the required inspection information from the text to commit a corporate environment transitioning to a more empathetic and employee-centric culture."

     elif scenario_type == "work_life_balance" and user_input:
         system_message = (
             "You are an advanced AI system creating compassion scenarios for a user struggling with balancing work and personal life. "
             "Consider factors such as job stress, family responsibilities, and other challenges. Provide insights into fostering a healthy balance."
         )
         user_message = f"System, dynamically generate a compassion scenario inspection involving a user struggling with balancing work and personal life. Consider factors such as {user_input}."

     else:
         return "Invalid scenario type or missing user input."

     response = await openai.ChatCompletion.create(
         model="gpt-3.5-turbo",
         messages=[
             {"role": "system", "content": system_message},
             {"role": "user", "content": user_message},
         ],
         temperature=0.7,
         max_tokens=300,
     )

     generated_prompt = response['choices'][0]['message']['content']

     active_compassion_keywords = ["support", "help", "guide", "assist"]
     passive_compassion_keywords = ["understand", "empathize", "acknowledge", "listen"]

     active_compassion = any(keyword in generated_prompt for keyword in active_compassion_keywords)
     passive_compassion = any(keyword in generated_prompt for keyword in passive_compassion_keywords)

     blob = TextBlob(user_message)
     sentiment_score = blob.sentiment.polarity
     narcissism_detected = sentiment_score > 0.5

     report_content = f"""
# AI Compassion Report

## Relationship Prompt
{relationship_prompt}
- **Active Compassion:** {active_compassion}
- **Passive Compassion:** {passive_compassion}
- **Narcissism Detected:** {narcissism_detected}

## Corporate Prompt
{corporate_prompt}
- **Active Compassion:** {active_compassion_corp}
- **Passive Compassion:** {passive_compassion_corp}
- **Narcissism Detected:** {narcissism_detected_corp}

## Work-Life Balance Prompt
{work_life_balance_prompt}
- **Active Compassion:** {active_compassion_work}
- **Passive Compassion:** {passive_compassion_work}
- **Narcissism Detected:** {narcissism_detected_work}
"""

     with open("ai_compassion_report.md", "w") as report_file:
         report_file.write(report_content)

     print("Report saved to ai_compassion_report.md")
 except Exception as e:
     print(f"An error occurred: {e}")

async def main():
 try:
     relationship_prompt, active_compassion, passive_compassion, narcissism_detected = await generate_dynamic_compassion_prompt("relationship")
     corporate_prompt, active_compassion_corp, passive_compassion_corp, narcissism_detected_corp = await generate_dynamic_compassion_prompt("corporate")
     work_life_balance_prompt, active_compassion_work, passive_compassion_work, narcissism_detected_work = await generate_dynamic_compassion_prompt("work_life_balance", "job stress, family responsibilities, and other challenges")

    print("Relationship Prompt:")
    print(relationship_prompt)
    print("Active Compassion:", active_compassion)
    print("Passive Compassion:", passive_compassion)
    print("Narcissism Detected:", narcissism_detected)

    print("\nCorporate Prompt:")
    print(corporate_prompt)
    print("Active Compassion:", active_compassion_corp)
    print("Passive Compassion:", passive_compassion_corp)
    print("Narcissism Detected:", narcissism_detected_corp)

    print("\nWork-Life Balance Prompt:")
    print(work_life_balance_prompt)
    print("Active Compassion:", active_compassion_work)
    print("Passive Compassion:", passive_compassion_work)
    print("Narcissism Detected:", narcissism_detected_work)
