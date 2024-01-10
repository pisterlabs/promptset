import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
model = "gpt-3.5-turbo-1106"
#model = "gpt-4"
directory = "/home/Cithoreal/Nextcloud/Documents/Audio Journals/Transcriptions/"
unprocessed = directory + "Unprocessed/"
processed = directory + "Processed/"
done = directory + "Done/"

def process_journal(transcription):

    abstract_summary = abstract_summary_extraction(transcription)
    action_items = action_item_extraction(transcription)
    events = events_extraction(transcription)
    
    return {
        'abstract_summary': abstract_summary,
        'action_items': action_items,
        'events': events
    }



def abstract_summary_extraction(transcription):
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content":  "You are a highly skilled AI trained in language comprehension and summarization, with a focus on personal and professional narratives. Please read the text and summarize it into a concise abstract paragraph. The summary should reflect my first-person perspective, capturing key points relevant to my personal life and professional project ideas, while omitting extraneous details."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response.choices[0].message.content



def action_item_extraction(transcription):
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are an AI expert in analyzing conversations for actionable insights. Review the text and identify tasks, assignments, or actions."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response.choices[0].message.content

def events_extraction(transcription):
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are an AI expert in analyzing conversations for actionable insights. Review the text and identify events mentioned."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response.choices[0].message.content



#Save as formatted .md file instead of docx
def save_as_md(minutes, filename):
    with open(filename, "w") as f:
        for key, value in minutes.items():
            # Replace underscores with spaces and capitalize each word for the heading
            heading = ' '.join(word.capitalize() for word in key.split('_'))
            f.write(f"# {heading}\n")
            f.write(f"{value}\n\n")



#Loop through each file in the directory and transcribe it, when finished move the file to the processed folder
for filename in os.listdir(unprocessed):
    if filename.endswith(".txt"):
        print(filename)
        transcription = open(unprocessed + filename, "r").read()
        document = process_journal(transcription)
        os.rename(unprocessed + filename, done + filename)
        save_as_md(document, processed + filename[:-3] + ".md")
    else:
        continue