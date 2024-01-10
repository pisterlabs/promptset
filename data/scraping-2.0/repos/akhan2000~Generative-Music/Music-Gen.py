import openai
from midiutil import MIDIFile
import random

# Define scales for different genres
scales = {
    'minor': [0, 2, 3, 5, 7, 8, 10],  
    'major': [0, 2, 4, 5, 7, 9, 11],
    'blues': [0, 3, 5, 6, 7, 10],
    'jazz': [0, 2, 4, 6, 7, 9, 11],
    'rock': [0, 2, 4, 5, 7, 9, 11],
    'classical': [0, 2, 4, 5, 7, 9, 11],
    'pop': [0, 2, 4, 5, 7, 9, 11],  
}

start_pitches = {
    'C': 60, 'C#': 61, 'Db': 61, 'D': 62, 'D#': 63, 'Eb': 63,
    'E': 64, 'F': 65, 'F#': 66, 'Gb': 66, 'G': 67, 'G#': 68, 'Ab': 68,
    'A': 69, 'A#': 70, 'Bb': 70, 'B': 71
}


def get_music_attributes_from_text(text, api_key):
    openai.api_key = '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'
    try:
        prompt = (
            "The following is a user's description of a melody they want to create: '{}'. "
            "Based on this description, provide detailed musical attributes including the genre, key, and BPM. "
            "The program uses musical scales like minor, major, blues, jazz, rock, classical, and pop, and supports all musical keys from A0 to G#0/Bb0."
            "For example, if the description is 'a fast-paced rock melody in the key of E', the response should be 'genre: rock, key: E, BPM: 140'."
            "\n\nAnalyze and describe the musical attributes for this description:"
        ).format(text)

        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=100
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def create_melody(scale_notes, length=32):
    melody = []
    for _ in range(length):
        note = random.choice(scale_notes)
        duration = random.choice([1, 0.5, 0.25, 0.75])  # Including triplet feel
        dynamics = random.choice(range(70, 110))  # Varying dynamics
        melody.append((note, duration, dynamics))
    return melody

def create_midi_from_attributes(attributes, file_name="output.mid"):
    midi_file = MIDIFile(1)
    track = 0
    time = 0
    midi_file.addTempo(track, time, attributes['bpm'])

    genre_scale = scales.get(attributes['genre'], scales['major'])
    key = attributes.get('key', 'C')
    start_pitch = start_pitches.get(key, 60)
    scale_notes = [start_pitch + interval for interval in genre_scale]

    melody = create_melody(scale_notes)

    for note, duration, dynamics in melody:
        midi_file.addNote(track, 0, note, time, duration, dynamics)
        time += duration

    with open(file_name, "wb") as output_file:
        midi_file.writeFile(output_file)
    print(f"MIDI file {file_name} has been created")    

def main():
    api_key = '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'
    user_input = input("Enter your music description (e.g., 'I want a happy pop melody at 90 bpm in key of E'): ")
    attributes_text = get_music_attributes_from_text(user_input, api_key)

    if attributes_text:
        attributes_list = attributes_text.lower().split()
        attributes = {
            'genre': next((word for word in attributes_list if word in scales), 'major'),
            'bpm': int(next((attributes_list[i-1] for i, word in enumerate(attributes_list) if word == 'bpm'), 120)),
            'key': next((word for word in attributes_list if word in start_pitches), 'C'),
            # Additional attribute parsing can be added here
        }
        create_midi_from_attributes(attributes)
    else:
        print("No response was received from the API.")

if __name__ == "__main__":
    main()




# import openai
# from midiutil import MIDIFile
# import random

# def get_music_attributes_from_text(text, api_key):
#     openai.api_key = api_key

#     try:
#         response = openai.Completion.create(
#             model="gpt-3.5-turbo-instruct",
#             prompt=text,
#             max_tokens=50
#         )
#         content = response.choices[0].text.strip()
#         return content
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None

# def create_midi_from_attributes(attributes, file_name="output.mid"):
#     midi_file = MIDIFile(1)
#     track = 0
#     time = 0
#     midi_file.addTempo(track, time, attributes['bpm'])

#     start_pitch = 57 if attributes['mood'] == 'dark' else 60
#     scale = [0, 2, 3, 5, 7, 8, 10] if attributes['mood'] == 'dark' else [0, 2, 4, 5, 7, 9, 11]
#     scale_notes = [start_pitch + interval for interval in scale]

#     for _ in range(16):
#         change = random.choice([-1, 0, 1])
#         duration = random.choice([1, 0.5, 0.25])

#         current_pitch = scale_notes[random.randint(0, len(scale_notes) - 1)] if change == 0 else max(0, min(scale_notes[-1] + change, 127))
#         if current_pitch in scale_notes:
#             midi_file.addNote(track, 0, current_pitch, time, duration, 100)
#             time += duration

#     with open(file_name, "wb") as output_file:
#         midi_file.writeFile(output_file)
#     print(f"MIDI file {file_name} has been created")    

# def main():
#     api_key = 'your-api-key'  # Replace with your actual OpenAI API key
#     user_input = input("Enter your music description (e.g., 'I want a dark 120 bpm techno melody'): ")
#     attributes_text = get_music_attributes_from_text(user_input, api_key)

#     if attributes_text:
#         attributes_list = attributes_text.lower().split()
#         attributes = {
#             'mood': 'dark' if 'dark' in attributes_list else 'bright',
#             'bpm': int(attributes_list[attributes_list.index('bpm') - 1]) if 'bpm' in attributes_list else 120
#         }
#         create_midi_from_attributes(attributes)
#     else:
#         print("No response was received from the API.")

# if __name__ == "__main__":
#     main()
