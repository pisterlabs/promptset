from langchain.text_splitter import Document
from youtube_helper import create_metadata

def split_transcript(transcript, word_limit):
    transcript_text = transcript.fetch()
    parts = []
    current_content = []
    current_word_count = 0

    start_time = None
    for entry in transcript_text:
        words = entry['text'].split()
        new_word_count = current_word_count + len(words)

        if start_time is None:
            start_time = entry['start']

        if new_word_count < word_limit:
            current_content.append(f"\n+{entry['start']} + \t{entry['text'].strip()}")
            current_word_count = new_word_count
        else:
            end_time = entry['start']
            metadata = create_metadata(transcript, start_time, end_time)
            parts.append(Document(page_content=''.join(current_content), metadata=metadata))
            current_content = [f"\\n+{entry['start']} + \t{entry['text'].strip()}"]
            current_word_count = len(words)
            start_time = entry['start']

    if current_content:
        end_time = transcript_text[-1]['start']
        metadata = create_metadata(transcript, start_time, end_time)
        parts.append(Document(page_content=''.join(current_content), metadata=metadata))
    return parts
