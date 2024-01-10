#Correcting input transcript
from openai import OpenAI
import os

def process_string(input_string):
    # Split the input string into lines
    lines = input_string.split('\n')

    # Process each line
    for i in range(len(lines)):
        # Remove the timestamp
        words = lines[i].split(' ', 1)
        if len(words) > 1:
            lines[i] = words[1]

        # Remove opening brackets
        lines[i] = lines[i].replace('[', '')

        # Replace closing brackets with a colon
        lines[i] = lines[i].replace(']', ':')

    # Join the processed lines back into a string
    result_string = '\n'.join(lines)

    return result_string

def combine_lines(text):
    lines = text.split("\n")
    combined_lines = []
    current_speaker = ""
    current_text = ""

    for line in lines:
        if not line.strip():
            continue

        speaker, sentence = line.split(":", 1)
        if speaker == current_speaker:
            current_text += " " + sentence.strip()
        else:
            if current_text:
                combined_lines.append(f"{current_speaker}: {current_text}")
            current_speaker = speaker
            current_text = sentence.strip()

    # Add the last speaker's text
    if current_text:
        combined_lines.append(f"{current_speaker}: {current_text}")

    return "\n".join(combined_lines)

def correctText(fixed_output):
    
    fixed_output = process_string(fixed_output)
    fixed_output = combine_lines(fixed_output)

    client = OpenAI(
        api_key="",
    )

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Correct any grammar and spelling mistakes in the user dialogue transcript."},
            {"role": "user", "content": fixed_output}
        ]
    )


    fixed_output = completion.choices[0].message.content


    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Which speaker is the doctor? Answer in one word."},
            {"role": "user", "content": fixed_output}
        ]
    )
    doctor = completion.choices[0].message.content




    if "SPEAKER_00" in doctor:
        fixed_output = fixed_output.replace("SPEAKER_00", "Doctor")
        fixed_output = fixed_output.replace("SPEAKER_01", "Patient")
        # docGender = sp0
        # patGender = sp1
    else:
        fixed_output = fixed_output.replace("SPEAKER_01", "Doctor")
        fixed_output = fixed_output.replace("SPEAKER_00", "Patient")
        # docGender = sp1
        # patGender = sp0

    print(fixed_output)

    # Combine title and conversation
    full_text = 'Conversation Script' + "\n\n" + fixed_output

    # File name
    file_name = "conversation_script.txt"

    # Writing to file
    with open(os.path.join('outputs',file_name), "w") as file:
        file.write(full_text)

    print(f"The conversation script has been saved as '{file_name}'.")

