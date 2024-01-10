import os
import openai
import pandas as pd

openai.api_key = "sk-dummykey"

def get_transcription(audio_file):
    with open(audio_file, "rb") as file:
        #transcription = {"text": "test"}
        transcription = openai.Audio.transcribe("whisper-1", file)
    total_text = transcription["text"]
    #total_text = " ".join([segment["text"] for segment in transcription['segments']])
    return total_text

def main(args):
    # Load the existing CSV file or initialize a new one
    csv_path = os.path.join(args.folder, 'transcriptions.csv')
    if os.path.isfile(csv_path):
        transcriptions_df = pd.read_csv(csv_path)
    else:
        transcriptions_df = pd.DataFrame(columns=['id', 'transcription'])

    # Get the set of existing ids
    existing_ids = set(transcriptions_df['id'])

    # Process each audio file in the folder
    for filename in os.listdir(args.folder):
        if filename.endswith(".wav") or filename.endswith(".mp3"):  # or whatever your audio file extensions are
            print(f"Processing {filename}")
            # Extract id from the filename
            file_id = int(filename.split('.')[0])

            # Skip the file if its id already exists in the csv
            if file_id in existing_ids:
                continue

            # Transcribe the file and add the transcription to the DataFrame
            file_path = os.path.join(args.folder, filename)
            transcription = get_transcription(file_path)
            new_row = pd.DataFrame({
                'id': [file_id],
                'transcription': [transcription]
            })
            transcriptions_df = pd.concat([transcriptions_df, new_row], ignore_index=True)


    # Save the updated DataFrame to the CSV file
    print("Writing csv file")
    transcriptions_df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=str, required=True, help="Folder containing audio files")
    args = parser.parse_args()
    main(args)

