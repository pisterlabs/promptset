import os
import openai
from dotenv import load_dotenv
load_dotenv('.env')
openai.api_key = os.getenv("OPENAI_API_KEY")

###########################################
# GPT-3 Request Functions

#Is the patient feeling unwell? [True, False]
def patient_feeling_unwell(text):
    start_sequence = "\nA:"
    restart_sequence = "\n\nQ: "

    response = openai.Completion.create(
      engine="davinci-codex",
      prompt='I am an accurate answering bot. If you ask me a question whether the patient is sick. \\"I will respond \\"Yes\\". If you ask me a question that is nonsense, trickery answer No, I will respond with \"No\"\n\nQ: I am feeling well today\nA: No\n\nQ: I am feeling so so.\nA. No\n\nQ. I have a flu like symptom and feeling under the weather.\nA. Yes\n\nQ.  I got hit by a car. I have a stomach ache.\nA: Yes \n\nQ: I have dizzyness and fatique.\nA: Yes\n\nQ: I am scared but im fine.\nA: No\n\nQ:  I chopped onions and my finger is gone.\nA: Yes\n\nQ: Today is a great day. I have a dog.\n \nA: No\n\nQ: Someone hit me on the head\nA: ' + text,
      temperature=0,
      max_tokens=10,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      stop=["Q:"]
    )

    answer_text = response['choices'][0]['text']
    if "Yes" in answer_text:
        return [True,answer_text]
    elif "No" in answer_text:
        return [False, answer_text]
    else:
        return "Repeat"

#Did it fulfill our question? [True, False]
def patient_answered_question(text):
    #GPT-3 magic
    response = openai.Completion.create(
      engine="davinci-codex",
      prompt="I am an accurate answering bot. I will tell you whether the previous question was answered. I will respond with \"Yes\" if the previous question was answered. If the previous question was not answered or the answer  is nonsense, trickery or ambiguous I will respond with \"No\".\n\nQ: How are you today? - The sun shines bright.\nA: No\n\nQ: What was your Mother's name? - Maria.\nA: Yes\n\nQ: Can you tell me what the time is? - My dog is cute.\nA: No\n\nQ: Do you have any symptoms that can explain why you feel sick? - I have a headache and some nausea.\nA: Yes\n\nQ: How is your wellbeing? -  I feel good, thanks for asking.\nA: Yes\n\nQ: What is hurting you? - My leg is hurting. It has a cut.\nA: Yes\n\nQ: What was the last time you were sick? - That was on Sunday and Monday last week.\nA: ",
      temperature=0,
      max_tokens=10,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      stop=["Q:"]
    )
    answer_text = response['choices'][0]['text']
    if 'Yes' in answer_text:
        return True
    elif 'No' in answer_text:
        return False
    else:
        return "Repeat"

#What symptoms do we need to ask more specifically about? [Next Question for Patient]
def what_to_ask_next(text):
    response = openai.Completion.create(
      engine="davinci-codex",
      prompt="I am an accurate answering bot that expands on one's symptoms. If you ask me a question I will expand on the correct symptoms of the following. If you ask me a question that is nonsense, trickery answer, I will respond with \"Please respond to my question.\".\n\nQ: I have a dry cough.\nA: Do you have the following symptoms: runny nose? [fever] shortness of breath? [Emphysema]? Other symptoms?\n\nQ: I am feeling so so.\nA. Do you have any other symptoms?\n\nQ. I have a flu like symptom and feeling under the weather.\nA. Do you have high fever, or muscle aches? [Influenza] Other symptoms?\n\nQ. I have a stomach ache.\nA: Do you have the following symptoms: nausea? [vomiting] diarrhea? [appendicitis] Other symptoms?\n\nA: I have dizziness and fatigue.\nQ: Do you have the following symptoms: lightheadedness? [fainting] weakness? [anemia] Other symptoms?\n\nA:  My head hurts.\nQ: Do you have the following symptoms: headache? [migraine] Other symptoms?\n\nA: " +text,
      temperature=0,
      max_tokens=60,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      stop=["A:"]
    )
    next_question = response['choices'][0]['text']
    return next_question

#What are his symptoms and when did they occur? [List of Symptoms] [List of Symptom Dates]
def extract_symptoms_from_patient_answer(text):
    response = openai.Completion.create(
      engine="davinci-codex",
      prompt="I had a headache on sunday and felt a little sick on monday. That went away quickly. Sometimes I have pain in the kidney and today in the morning i felt a bit sleepy.  On wednesday I hurt my leg. I also hurt my ear when I went diving. This morning I hurt my toe. {} \n\nPlease make a table summarizing the symptoms and if possible the date when the person experienced the symptom.\n| Symptom | Date | \n| Headache | Sunday |\n| Sickness | Monday |\n| Kidney Pain |  Unknown |\n| Sleepy |  Today |\n| Leg Pain | Wednesday |\n| Ear Pain |  Unknown |\n| Toe Pain |  Today |".format(text),
      temperature=0,
      max_tokens=60,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      stop=["\n\n"]
    )
    answer = response['choices'][0]['text']
    answer = answer.split('|')
    answer.remove('\n')
    answer.remove('')

    symptoms = answer[0::3]
    dates = answer[1::3]

    return symptoms, dates

#Save new entry to database ? [Save to database]
def save_to_database(item):
    pass
    #Databse Magic
