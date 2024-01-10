import os
import openai
from youtube_transcript_api import YouTubeTranscriptApi


# set openai api key
openai.api_key = os.getenv("chat_gpt_api_key")


def getTranscript(videoId):
    # assigning srt variable with the list
    # of dictionaries obtained by the get_transcript() function
    srt = YouTubeTranscriptApi.get_transcript(videoId)

    return "\n".join([s["text"] for s in srt])


# summerize text with chatgpt
def summarize_text(text):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"""The following is a section of the transcript of a youtube video:
        {text}
        Summarize this section of the transcript.""",
            max_tokens=500,
            temperature=0.2,
            frequency_penalty=0,
        )
        return response.choices[0].text
    except:
        return None


def get_video_summary(video_id):
    captions = getTranscript(video_id)
    if captions:
        summary = summarize_text(captions)
        return summary
    return "No captions found for this video."


#
#
# diagnostics = 0
# include_mentions = 0
#
#
# def get_chunks_from_youtube(video_id):
#     transcript = YouTubeTranscriptApi.get_transcript(video_id)
#     chunks = []
#     start_timestamp = 0.0
#     current_timestamp_mins = 0.0
#     current_chunk = []
#     for entry in transcript:
#         current_timestamp_mins = entry["start"] / 60.0
#         if current_timestamp_mins - start_timestamp > 10:
#             chunks.append(current_chunk)
#             start_timestamp = current_timestamp_mins
#             current_chunk = []
#
#         current_chunk.append(entry["text"])
#
#     if len(current_chunk) > 0:
#         chunks.append(current_chunk)
#
#     print(f"Found {len(chunks)} chunks")
#
#     return chunks
#
#
# def summarize_chunk(index, chunk):
#     chunk_str = "\n".join(chunk)
#     prompt = f"""The following is a section of the transcript of a youtube video. It is section #{index+1}:
#     {chunk_str}
#     Summarize this section of the transcript."""
#
#     if diagnostics:
#         for line in prompt.split("\n"):
#             print(f"# {line}")
#
#     completion = openai.Completion.create(
#         engine="text-davinci-003",
#         max_tokens=500,
#         temperature=0.2,
#         prompt=prompt,
#         frequency_penalty=0,
#     )
#
#     msg = completion.choices[0].text
#
#     if diagnostics:
#         print(f"# Response: {msg}")
#
#     return msg
#
#
# def summarize_the_summaries(summaries):
#     summaries_str = ""
#     for index, summary in enumerate(summaries):
#         summaries_str += f"Summary of chunk {index+1}:\n{summary}\n\n"
#
#     prompt = f"""The following are summaries of a youtube video in 10 minute chunks:"
#     {summaries_str}
#     Summarize the summaries."""
#
#     if diagnostics:
#         # print each line of the prompt with a leading # so we can see it in the output
#         for line in prompt.split("\n"):
#             print(f"# {line}")
#
#     completion = openai.Completion.create(
#         engine="text-davinci-003",
#         max_tokens=500,
#         temperature=0.2,
#         prompt=prompt,
#         frequency_penalty=0,
#     )
#
#     msg = completion.choices[0].text
#
#     if diagnostics:
#         print(f"# Response: {msg}")
#
#     return msg
