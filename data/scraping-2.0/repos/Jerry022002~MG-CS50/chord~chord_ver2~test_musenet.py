import openai

# Set up your OpenAI API key
openai.api_key = "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"

# Define input parameters
chords = "Cmaj7, Dm7, G7, Cmaj7"  # Example chord progression
tempo = "120"  # BPM
time_signature = "4/4"  # Time signature
instrument = "piano"  # Instrument type

# Generate music using MuseNet
prompt = f"Generate music in {time_signature} time signature, at {tempo} BPM, using {instrument}, based on the chord progression: {chords}"
response = openai.Completion.create(
    engine="text-davinci-003",  # Choose the appropriate engine
    prompt=prompt,
    max_tokens=200  # Adjust the length of the generated music
)

# Extract the generated music from the response
generated_music = response.choices[0].text.strip()

# Save the generated music to a MIDI file or use it as needed
with open("generated_music.mid", "w") as f:
    f.write(generated_music)
