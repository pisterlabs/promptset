import openai
import json
import nltk

def segment_text(text, words_per_segment):
    sentences = nltk.sent_tokenize(text)
    segments = []
    current_segment = ''

    for sentence in sentences:
        if len(current_segment.split()) + len(nltk.word_tokenize(sentence)) <= words_per_segment:
            current_segment += ' ' + sentence
        else:
            segments.append(current_segment.strip())
            current_segment = sentence

    if current_segment:
        segments.append(current_segment.strip())

    return segments

API_KEY = '[openai_api_key]'
model_id = 'whisper-1'

media_file_path = '/content/testmp3.mp3'
media_file = open(media_file_path, 'rb')

response = openai.Audio.transcribe(
    api_key=API_KEY,
    model=model_id,
    file=media_file
)

responsetext = response["text"]

segmented_text = segment_text(responsetext, 45)

print(segmented_text)
