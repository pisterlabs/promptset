import os
import json
import openai

from langchain.model_io.output_parsers.structured import OutputFixingParser

# Initialize GPT-4 API
openai.api_key = os.environ.get("OPENAI_KEY")

# Initialize Langchain OutputFixingParser
parser = OutputFixingParser()

def gpt4_summary(request):
    # Get annotated transcript from request
    annotated_transcript = request.get_json().get('annotated_transcript')

    # GPT-4 Prompt with Business Analyst persona
    prompt = f"""Persona: Business Analyst
    As a Business Analyst, my role is to provide clear, factual, and structured summaries of customer interactions.
    I need to analyze the annotated transcript below and generate a summary in a structured JSON format.
    
    The summary should strictly adhere to the following sections:

    1. Customer ID
    2. Participants IDs
    3. Source of Interaction
    4. Interaction Summary
    5. Sentiment Analysis for Each Participant
    6. Pertinent Personal or Business Information for Each Participant

    Example JSON Format:
    {
        "Summary": {
            "CustomerID": "XXX",
            "Source": "XXX"
        },
        "Participants": {
            "Participant1": {
            "ID": "XX",
            "Sentiment": "Positive",
            "PertinentInfo": "Relevant business or personal details"
            },
            "Participant2": {
            "ID": "XX",
            "Sentiment": "Neutral",
            "PertinentInfo": "Relevant business or personal details"
            }
        },
        "Interaction": {
            "Summary": "A detailed summary of the interaction, preserving full names, proper nouns like company names, agreed-upon next steps, mentioned dates and dates of interactions, and any significant business metrics.",
            "NextSteps": "List of agreed-upon next steps with corresponding dates. Dates should be written in YYYY-MM-DD format. Infer the year from the context of the date of the interaction."
        }
    }

    Note: The summary should be factual and based solely on the content of the transcript. Do not include any additional context or information not present in the transcript.

    Annotated Transcript:
    {annotated_transcript}
    """

    # Make GPT-4 API call
    response = openai.Completion.create(
        engine="gpt-4",
        prompt=prompt,
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.1
    )

    # Extract and parse GPT-4 output
    gpt4_output = response.choices[0].text.strip()
    parsed_output = parser.parse(gpt4_output)

    # Convert parsed output to JSON
    output_json = json.dumps(parsed_output)

    return output_json
