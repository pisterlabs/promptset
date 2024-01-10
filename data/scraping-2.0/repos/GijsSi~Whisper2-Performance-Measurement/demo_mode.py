import os
import sounddevice as sd
import wavio
from openai import OpenAI
import glob
import pandas as pd
import re
import json

# Function to send audio to Whisper API
def send_to_whisper_api(filepath):
    client = OpenAI()  # Initialize the OpenAI client

    
    print(" 🟢 : Sending the data...")

    with open(filepath, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            response_format="text"
        )
    return response


# Function to calculate Levenshtein distance
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

# Function to calculate Word Error Rate (WER)
def word_error_rate(s1, s2):
    b = set(s1.split() + s2.split())
    word2char = dict(zip(b, range(len(b))))

    w1 = [chr(word2char[w]) for w in s1.split()]
    w2 = [chr(word2char[w]) for w in s2.split()]

    return levenshtein_distance(''.join(w1), ''.join(w2)) / len(w2)


# Function to calculate Recall Rate
def recall_rate(original, transcribed):
    original_words = set(original.split())
    transcribed_words = set(transcribed.split())

    correct_words = original_words.intersection(transcribed_words)
    return len(correct_words) / len(original_words)

# Main function to process audio files and evaluate performance
def process_audio_files(directory, sentences):
    data = []
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    audio_files = glob.glob(directory + "/*.wav") + glob.glob(directory + "/*.mp3")

    for filepath in audio_files:
        filename = os.path.basename(filepath)

        try:
            transcript_text = send_to_whisper_api(filepath)
            if not transcript_text:
                continue

            match = re.search(r'\d+', filename)
            file_id = int(match.group()) if match else None
            if file_id is None or file_id > len(sentences):
                continue
            original_text = sentences[file_id - 1]

            wer = word_error_rate(original_text, transcript_text)

            # Calculate precision and recall
            transcribed_words = set(transcript_text.split())
            original_words_set = set(original_text.split())
            correct_words = original_words_set.intersection(transcribed_words)
            true_positives = len(correct_words)
            false_positives = len(transcribed_words - original_words_set)
            false_negatives = len(original_words_set - transcribed_words)

            total_true_positives += true_positives
            total_false_positives += false_positives
            total_false_negatives += false_negatives

            precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0

            # Append data for each file
            data.append({
                "Filename": filename,
                "Original Sentence": original_text,
                "Transcribed Sentence": transcript_text,
                "Word Error Rate": wer,
                "Levenshtein Distance": levenshtein_distance(original_text, transcript_text),
                "Precision": precision,
                "Recall": recall
            })

        except Exception as e:
            print(f"An error occurred processing {filename}: {e}")
    
    # Calculate micro averages
    if total_true_positives + total_false_positives > 0:
        micro_precision = total_true_positives / (total_true_positives + total_false_positives)
    else:
        micro_precision = 0

    if total_true_positives + total_false_negatives > 0:
        micro_recall = total_true_positives / (total_true_positives + total_false_negatives)
    else:
        micro_recall = 0

    if micro_precision + micro_recall > 0:
        micro_fscore = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
    else:
        micro_fscore = 0

    # Calculate macro averages
    num_data_points = len(data)
    macro_precision = sum(item.get('Precision', 0) for item in data) / num_data_points if num_data_points > 0 else 0
    macro_recall = sum(item.get('Recall', 0) for item in data) / num_data_points if num_data_points > 0 else 0
    macro_fscore = sum(2 * (item.get('Precision', 0) * item.get('Recall', 0)) / (item.get('Precision', 0) + item.get('Recall', 0)) if (item.get('Precision', 0) + item.get('Recall', 0)) > 0 else 0 for item in data) / num_data_points if num_data_points > 0 else 0

    # Append the averages to the data
    data.append({
        "Micro Precision": micro_precision,
        "Micro Recall": micro_recall,
        "Micro F-score": micro_fscore,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F-score": macro_fscore
    })
    return data


# Assuming the audio files are in a directory named 'audio_files'
audio_directory = 'wav'

# Read the transcript file
with open('transcript.txt', 'r') as file:
    sentences = [line.strip().split(': ', 1)[1] for line in file if ': ' in line]

# Process the audio files and generate reports
reports = process_audio_files(audio_directory, sentences)

# Convert the reports to a DataFrame and save to an Excel file
df_results = pd.DataFrame(reports)
df_results.to_excel("results.xlsx")
print(df_results)