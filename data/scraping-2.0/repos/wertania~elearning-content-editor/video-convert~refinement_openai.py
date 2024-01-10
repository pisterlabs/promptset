import json
from openai import OpenAI
from config import OPENAI_API_KEY, transscription_base_path
from video_types import Transcript
from logging_output import info, error, warning, debug
import os


client = OpenAI(api_key=OPENAI_API_KEY)


# read file custom_context.json if existing
phrases = []
# read context from file ".custom_context.json"
if os.path.exists(".custom_context.json"):
    debug("Found custom context file")
    with open(".custom_context.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        phrases = data["phrases"]
# phrases as comma separated list
phrases_cs = ", ".join(phrases)
debug("Phrases: " + phrases_cs)

prompt = """
Du bist ein Mitarbeiter der Firma Hochhuth und machst Videos zum Training von Softwaremodulen der Firma Hochhuth.
Du dienst als automatische Korrektursoftware für Transkriptionen aus den Videos.
Deine Aufgabe ist es, die übermittelten Sätze zu korrigieren und in eine logische und formelle Form zu überführen.
Vermeide dabei Umgangssprache und unpräzise Ausdrucksformen. Du machst die Sätze dabei nicht unnötig länger.
Es gibt einige Markennamen, die nicht verändert werden dürfen, oder die gegebenenfalls im Transkript falsch interpretiert wurden.
Diese sind: {phrases_cs}
Antworte im JSON Format exakt wie in der Eingabe, und ersetze lediglich die Inhalte innerhalb von "text": "...".
"""


def refine(transcript: Transcript, guid: str) -> Transcript:
    debug("Refining transcript...")

    print(transcript)

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": transcript.to_json(),
            },
        ],
    )
    response = completion.choices[0].message.content

    if response is None:
        raise Exception("OpenAI returned no response")

    refined_transcript = Transcript.from_json(response)

    # merge sentences with same start time
    # sometimes openai splits sentences with the same start time into multiple sentences
    merged_sentences = []

    for sentence in refined_transcript.sentences:
        if merged_sentences and merged_sentences[-1].start_time == sentence.start_time:
            merged_sentences[-1].text += " " + sentence.text
        else:
            merged_sentences.append(sentence)

    refined_transcript.sentences = merged_sentences

    # save refined transcript to file
    debug(
        "Saving refined transcript to file "
        + transscription_base_path
        + guid
        + ".openai.json"
    )
    with open(
        transscription_base_path + guid + ".openai.json", "w", encoding="utf-8"
    ) as f:
        json.dump(refined_transcript.to_dict(), f, indent=4, ensure_ascii=False)

    return refined_transcript
