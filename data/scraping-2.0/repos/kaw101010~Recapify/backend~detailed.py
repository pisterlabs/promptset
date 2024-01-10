from transcripts_test import *
import openai


def detailed_notes(transcript):
    openai.api_key = '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'  # Replace with your OpenAI API key
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Can you write me a detailed notes from the following transcription?:\n'{transcript}'",
        max_tokens=1000,  # Adjust token limit as needed
        temperature=0.5,  # Adjust the creativity of the response
        #stop=["\n\n"]  # Stop the generation at double newlines for better formatting
    )

    return response['choices'][0]['text']


