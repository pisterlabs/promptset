import os
import openai
import sys

openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the categories in the medical note
categories = [
    "Assessments",
    "Discussion",
    "History of Present Illness",
    "Plan"
]

# Define the prompt
prompt = "Put this transcription into a medical note. Note that this transcription is a conversation between a doctor and the patient, thus any mention of \"my, me, or I\" should not be ascribed to the patient's case. It should have the following categories, if they are discussed:\n\n" + "\n".join(categories) + "\n\n. The assessment section should be an ordered list of findings, labeled 1,2,3 etc. Do not include the word [REDACTED]. In the list of assessments, list the corresponding diagnoses according to ICD-10 guidelines. Provide not only the ICD-10 code, but also the definition, i.e. the name of the diagnosis. Write the plan section in a list of bullet points. The discussion section should be free text other relevant details to the patient's case. The account number may be stated in the beginning of the recording. If it is, label the note at the top with the account number"

# Get the output of the previous script
input_text = sys.stdin.read().strip()

# Add the transcription text to the prompt
prompt += input_text

# Send the prompt to the GPT-4 API
response = openai.ChatCompletion.create(
    model="gpt-4",  # This might change when GPT-4 is released
    messages=[{
        "role":"user", "content":prompt
    }]
)

print(response['choices'][0]['message']['content'].strip())
