import os
import openai
import argparse

openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the categories in the medical note
categories = [
    "Chief Complaints",
    "ROS (Review of symptoms)",
    "Medical history",
    "Surgical history",
    "Hospitalization/major diagnostic procedures",
    "Family history",
    "Social history",
    "Medications",
    "Allergies",
    "Assessment - Discussion",
    "Assessment - Plan",
    "Assessment - Plan - Treatment",
    "Assessment - Plan - Preventative Medicine",
]

# Define the prompt
prompt = "Put this transcription into a medical note. Note that this transcription is a conversation between a doctor and the patient, thus any mention of \"my, me, or I\" should not be ascribed to the patient's case. It should have the following categories, if they are discussed:\n\n" + "\n".join(categories) + "\n\n"

# Argument parser
parser = argparse.ArgumentParser(description='Convert transcriptions into medical notes.')
parser.add_argument('transcription_file', help='The path to the transcription file.')
args = parser.parse_args()

# Get the output of the previous script
with open(args.transcription_file, 'r') as f:
    input_text = f.read().strip()

# Add the transcription text to the prompt
prompt += input_text

# Call your local LLM model and feed the prompt
output_text = os.popen('./main -p \'{}\''.format(prompt)).read()

# Save the output to a file
with open(args.transcription_file.replace('_transcription.txt', '_medical_note.txt'), 'w') as f:
    f.write(output_text)
