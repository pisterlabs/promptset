import json
import os
from langchain import OpenAI, PromptTemplate
from google_cloud_utils import get_secret

OPENAI_API_KEY = get_secret('OPENAI_API_KEY')

# Template for the prompt for the speaker identification
speaker_id_template = """
What are the names of {speakers} from the following podcast transcript and description? 
Output as JSON in the following format:
{speakers_json_format}

Title:
"{title}"
Description:
"{description}"

Transcript:
"{transcript}"
"""


def run_llm_speaker_identification(transcript):
    """
    Do the speaker identification using LLM.
    If the speaker is not identified, then the speaker is set as Unknown.
    :param transcript: List of dialogs, speakers and information about the show
    :return: List of speakers identified matching the IDs from the transcript
    """

    # Create a string with the start of the podcast of the length of 1000 words
    start_transcript = ''
    t_speakers = transcript['speakers']

    speakers = set()
    for segment in transcript['dialogs']:
        if len(start_transcript.split()) < 500:
            if segment['text'] == '':
                continue
            start_transcript += t_speakers[segment['speaker']] + ':\n'
            start_transcript += segment['text'] + '\n'
            speakers.add(segment['speaker'])

    # narrow down the speakers to the ones that are in the start_transcript
    tt_speakers = [t_speakers[speaker] for speaker in speakers]

    # List of speakers from the transcript start written as SPEAKER_1, SPEAKER_2, ...
    speakers_str = ', '.join(tt_speakers)
    speakers_json_format = ', '.join(
        [speaker + '": "Name' + '"' for speaker in tt_speakers])
    speakers_json_format = '{' + speakers_json_format + '}'

    speaker_prompt = PromptTemplate(template=speaker_id_template,
                                    input_variables=['speakers', 'speakers_json_format', 'title', 'description',
                                                     'transcript'])
    speaker_prompt_text = speaker_prompt.format(speakers=speakers_str, speakers_json_format=speakers_json_format,
                                                title=transcript['title'],
                                                description=transcript['description'], transcript=start_transcript)

    print(speaker_prompt_text)
    # Create the OpenAI object
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    llm = OpenAI(temperature=0, model_name='gpt-3.5-turbo')
    llm_output = llm(speaker_prompt_text)
    print(llm_output)

    # Convert the llm_output JSON to a dictionary
    speakers_id = json.loads(llm_output)

    return speakers_id

