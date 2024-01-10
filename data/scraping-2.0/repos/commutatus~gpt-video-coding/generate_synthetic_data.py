import json
import openai
import os
import concurrent.futures
import openai.error
import time
import random

os.environ["OPENAI_API_KEY"] = "openAIKey"


def generate_conversation(max_retries = 3):
    focus = random.choice(focus_list)
    tone = random.choice(tone_list)
    style = random.choice(style_list)
    pace = random.choice(pace_list)
    structure = random.choice(structure_list)
    dynamics = random.choice(dynamics_list)
    intensity = random.choice(intensity_list)
    language = random.choice(language_list)
    guidance_level = random.choice(guidance_level_list)
    first_word = random.choice(first_word_list)
    second_word = random.choice(second_word_list)
    length = random.choice(length_list)
    messages = [
        {"role": "system", "content": f"""You are a therapy conversation generator. Your task is to generate a single therapy conversation that is as long as possible, based on parameters you will be given. Do not stop writing until it is impossible to continue."""},
        {"role": "user", "content": f"""Please generate a single therapy conversation with the following specifications:
        Focus: The main topic of the conversation should be {focus}.
        Tone: The overall emotional quality of the conversation should be {tone}.
        Style: The manner of expression in the conversation should be {style}.
        Pace: The rhythm or speed at which the conversation progresses should be {pace}.
        Structure: The organization or format of the conversation should be {structure}.
        Dynamics: The interaction pattern between the participants should be {dynamics}.
        Intensity: The emotional charge or depth of the conversation should be {intensity}.
        Language: The choice of words and phrases in the conversation should be {language}.
        Guidance Level: The degree to which the conversation is guided or directed by the therapist should be {guidance_level}.
        First message: The first message from the client must contain the words {first_word} and {second_word}.
        Therapist: The therapist, named Alex, should be curious, loving, empathetic, and use simple language. The therapist should ask the right questions.
        Client: The client, named Charlie, should write messages of {length} length and use simple language. The client is {tone}.
        Now generate the conversation, which should be as long as possible, starting with the client's first message which should include {first_word} and {second_word}, and then alternating between the client and the therapist in the following format:
        Client:
        Therapist:"""}
    ]

    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
            )
            return {
                "parameters": {
                    "focus": focus,
                    "tone": tone,
                    "style": style,
                    "pace": pace,
                    "structure": structure,
                    "dynamics": dynamics,
                    "intensity": intensity,
                    "language": language,
                    "guidance_level": guidance_level,
                    "first_word": first_word,
                    "second_word": second_word
                },
                "conversation": response['choices'][0]['message']['content']
            }

        except openai.error.OpenAIError as e:
            print(f'OpenAI error occurred: {e}, attempt {attempt + 1} of {max_retries}')
            time.sleep(1)
        
    print(f'Failed after {max_retries} attempts')
    return None


def process_file(output_file):
    start_time = time.time()

    # Open the file in 'a' mode
    with open(output_file, 'a') as out_file:
        # Create a ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
            # Create a dictionary to store the Future objects
            future_to_data = {executor.submit(generate_conversation): None for _ in range(70000)}

            for future in concurrent.futures.as_completed(future_to_data):
                try:
                    data = future.result()
                except Exception as exc:
                    print('Generated an exception: %s' % exc)
                else:
                    # Write the new conversation to the output file
                    out_file.write(json.dumps(data))
                    out_file.write('\n')  # JSONL files have one record per line
                    out_file.flush()

    end_time = time.time()

    print("Time taken: {} seconds".format(end_time - start_time))

focus_list = ["Personal relationships", "Work-related stress", "Family conflict", "Conflict with friends", "Heartbreak", "A Breakup", "Health issues", "Anxiety", "Depression", "Self-esteem", "Grief", "Addiction", "Dream come true", "New hobby", "Light small talk"]

tone_list = ["Calm", "Tense", "Hopeful", "Discouraged", "Anxious", "Upbeat", "Neutral", "Motivational", "Sad", "Depressed", "Happy", "Excited", "Loving"]

style_list = ["Formal", "Casual", "Directive", "Non-directive", "Solution-focused", "Client-centered", "Psychodynamic"]

pace_list = ["Slow and thoughtful", "Dynamic and energetic", "Steady and moderate", "Varying rhythm"]

structure_list = ["Question-answer format", "Free-flowing dialogue", "Guided reflection", "Structured interventions", "Advice"]

dynamics_list = ["Cooperative", "One-sided", "Equal contribution", "Therapist-guided"]

intensity_list = ["Light and surface-level", "Deep and emotionally charged", "Moderate and balanced", "Profoundly heartfelt and impactful", "Varying"]

language_list = ["Simple and straightforward", "Everyday language"]

guidance_level_list = ["Therapist-led", "Client-led", "Balanced and reciprocal"]

first_word_list = ["anxious", "depressed", "happy", "stressed", "worried", "excited", "nervous", "frustrated", "confused", "sad", 
                   "lonely", "overwhelmed", "angry", "afraid", "exhausted", "peaceful", "hopeful", "guilty", "friend",
                   "scared", "lost", "disappointed", "relieved", "unhappy", "tired", "embarrassed", "upset", "jealous", "numb", 
                   "insecure", "surprised", "disgusted", "grateful", "ashamed", "peace",
                    "shocked", "confident", "regretful", "loving", "defensive", "distracted", "painful"]

second_word_list = ["relationship", "mom", "girlfriend", "boyfriend", "family", "work", "health", "finances", "self-esteem", "friendship", "loss", "trauma", "trust",
                    "communication", "conflict", "stress", "change", "grief", "decision", "fear", "panic", "job", "him", "her", "he", "she", "i",
                    "addiction", "parenting", "isolation", "dependence", "failure", "success", "pressure", "responsibility", "childhood", "abuse",
                    "expectations", "bullying", "divorce", "marriage", "breakup", "neglect", "insecurity", "career", "school", "goals", "promise",
                    "dreams", "betrayal", "criticism", "control", "sexual", "rejection", "sorry", "loneliness", "cheat", "cheating", "fiancee", "wife", "husband"]

length_list = ["short", "long", "medium", "very short", "varying"]


output_file_path = ''

process_file(output_file_path)