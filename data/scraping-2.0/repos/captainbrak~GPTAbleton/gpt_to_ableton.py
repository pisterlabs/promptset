# Copyright by @BurnedGuitarist  |  https://www.youtube.com/@burnedguitarist/videos
# Draft script for processing GPT response and sending the MIDI notes into the Ableton clips with AbletonOSC and python-osc.

import os
import openai
import io
import re
import itertools
from sys import exit
from pythonosc import udp_client
import warnings

# Constants
PITCH = "pitch"
START_TIME = "start_time"
VELOCITY = "velocity"
DURATION = "duration"
SCHEMA = [PITCH, START_TIME, DURATION, VELOCITY]
VALID_MIDI_NOTES = range(30, 91)
SONG_TEMPO = 60

# AbletonOSC client parameters
IP = "127.0.0.1"
PORT = 11000

MELODY = "Melody"
DRUM = "Drum"
BASS = "Bass"

PROMPT = (
    "Human: Hi GPT, could you provide me with a simple longer melody written as a Ableton OSC pattern table? The table should have four columns: pitch, start_time, duration, velocity. Without any table boundaries. Columns should be separated by a single space, not tab. Rows should be separated by the new line character. Then in the second table write drums in the same format. Then in the third table write bass in the same format. Start time and duration should be expressed in Ableton OSC MIDI clip decimal format (e.g. 0.1). Notes should be in the Ableton OSC MIDO integer format. Velocity should be an integer, e.g. 127. All three tables should have the same sum of durations and be synchronized. Name the respective three table as Melody, Drum, Bass.",
)


def get_openai_response(prompt: str) -> str:
    """Get OpenAI API response for the user prompt."""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.9,
        max_tokens=350,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=[" Human:", " AI:"],
    )
    print(">>>>> raw response")
    print(response)
    return response["choices"][0]["text"]


def find_substring(string: str, start: str, end: str) -> str:
    """Find substring between keywords."""
    try:
        s = string.rindex(start) + len(start)
        e = string.rindex(end, s)
        return string[s:e]
    except ValueError:
        return ""


def parse_table_to_list(data: str, valid_midi_notes: list) -> list:
    """Convert text table to the array."""
    if data:
        data = [
            re.split("\t| ", row)
            for row in data.split("\n")
            if row and any(row.startswith(str(note)) for note in valid_midi_notes)
        ]
        data = [i for i in data if len(i) >= len(SCHEMA)]
        return data


def calculate_pattern_duration(data: list) -> int:
    """Calculate sum of clip durations."""
    t = float(data[0][SCHEMA.index(START_TIME)])
    for row in data:
        duration = float(row[SCHEMA.index(DURATION)])
        t += duration
    return round(t, 4)


def initiate_clips(
    ableton_client, song_tempo, melody_data, drum_data, bass_data
) -> None:
    """Initiate Ableton clips."""
    ableton_client.send_message("/live/song/set/tempo", song_tempo)

    melody_pattern_len = calculate_pattern_duration(melody_data)
    drum_pattern_len = calculate_pattern_duration(drum_data)
    bass_pattern_len = calculate_pattern_duration(bass_data)

    ableton_client.send_message("/live/clip_slot/delete_clip", (0, 0))
    ableton_client.send_message(
        "/live/clip_slot/create_clip", (0, 0, melody_pattern_len)
    )

    ableton_client.send_message("/live/clip_slot/delete_clip", (1, 0))
    ableton_client.send_message("/live/clip_slot/create_clip", (1, 0, drum_pattern_len))

    ableton_client.send_message("/live/clip_slot/delete_clip", (2, 0))
    ableton_client.send_message("/live/clip_slot/create_clip", (2, 0, bass_pattern_len))


def send_events(ableton_client, melody_data, drum_data, bass_data):
    """Send MIDI note data to the Ableton clips."""

    if melody_data:
        pointer = float(melody_data[0][SCHEMA.index(START_TIME)])
        for row in melody_data:
            pitch = int(row[SCHEMA.index(PITCH)])
            velocity = int(row[SCHEMA.index(VELOCITY)])
            duration = float(row[SCHEMA.index(DURATION)])
            ableton_client.send_message(
                "/live/clip/add/notes", (0, 0, pitch, pointer, duration, velocity, 0)
            )
            pointer += duration

    if drum_data:
        pointer = float(drum_data[0][SCHEMA.index(START_TIME)])
        for row in drum_data:
            pitch = int(row[SCHEMA.index(PITCH)])
            velocity = int(row[SCHEMA.index(VELOCITY)])
            duration = float(row[SCHEMA.index(DURATION)])
            ableton_client.send_message(
                "/live/clip/add/notes", (1, 0, pitch, pointer, duration, velocity, 0)
            )
            pointer += duration

    if bass_data:
        pointer = float(bass_data[0][SCHEMA.index(START_TIME)])
        for row in bass_data:
            pitch = int(row[SCHEMA.index(PITCH)])
            velocity = int(row[SCHEMA.index(VELOCITY)])
            duration = float(row[SCHEMA.index(DURATION)])
            ableton_client.send_message(
                "/live/clip/add/notes", (2, 0, pitch, pointer, duration, velocity, 0)
            )
            pointer += duration


if __name__ == "__main__":
    ableton_client = udp_client.SimpleUDPClient(IP, PORT)

    response_text = get_openai_response(PROMPT)

    if (
        MELODY not in response_text
        or DRUM not in response_text
        or BASS not in response_text
    ):
        warnings.warn("Incomplete GPT response. Try again.")
        exit()

    melody_data = parse_table_to_list(
        find_substring(response_text, MELODY, DRUM), VALID_MIDI_NOTES
    )
    drum_data = parse_table_to_list(
        find_substring(response_text, DRUM, BASS), VALID_MIDI_NOTES
    )
    bass_data = parse_table_to_list(response_text.split(BASS, 1)[1], VALID_MIDI_NOTES)

    if not melody_data or not drum_data or not bass_data:
        warnings.warn("Incorrect GPT response MIDI note format. Try again.")
        exit()

    initiate_clips(ableton_client, SONG_TEMPO, melody_data, drum_data, bass_data)

    send_events(ableton_client, melody_data, drum_data, bass_data)
