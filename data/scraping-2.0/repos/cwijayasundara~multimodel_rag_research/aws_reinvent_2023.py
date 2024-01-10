from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()


def extract_transcript():
    global f
    from youtube_transcript_api import YouTubeTranscriptApi
    # Replace 'VIDEO_ID_HERE' with the actual video ID
    video_id = 'PMfn9_nTDbME'
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        # Extract and print the transcript
        for entry in transcript:
            text = entry['text']
            with open('transcript.txt', 'a') as f:
                f.write(text + '\n')

    except Exception as e:
        print(f"An error occurred: {str(e)}")


# extract_transcript()

from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI

with open('transcript.txt', 'r') as f:
    docs = f.readlines()

print(docs)

llm = ChatOpenAI(temperature=0,
                 model_name="gpt-4-1106-preview")
