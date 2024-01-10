import os

from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers.audio import OpenAIWhisperParser, OpenAIWhisperParserLocal
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
#=========================================
#This is the draft with initial code
#=========================================

load_dotenv()

local = False

# urls = [f"https://youtu.be/{video_id}"]
urls = [f"https://youtu.be/jrSwgdiClc0"]

video_id = "jrSwgdiClc0"

package_dir = os.path.dirname(__file__)
sub_dir = "Audio"

# Directory where the audio will be saved
save_dir = os.path.join(package_dir, sub_dir)

loader = GenericLoader(YoutubeAudioLoader(urls, save_dir), OpenAIWhisperParser(api_key=os.environ.get("OPENAI_API_KEY").strip()))

docs = loader.load()

# print(docs)
# print(docs[0].page_content)

llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.environ.get("OPENAI_API_KEY").strip())

template = """
Summarize the following video transcript in one paragraph. Output your summary in json format, with 3 elements: "1_paragraph_summary", "similar_video_idea_summary" (here you should write 5-7 sentences on how a person can record a similar video) and a short (5-7 items) array of video tags/subjects relevant to the vieo idea, named "tags".
VIDEO: {transcript}
SUMMARY IN JSON FORMAT:
"""

prompt_template = PromptTemplate(
    input_variables=["transcript"],
    template=template
)

transcript = docs[0].page_content

prompt = prompt_template.format(transcript=transcript)

completion = llm(prompt)

print("prompt:", prompt)
print("completion:", completion)
with open(f"Youtube/{video_id}.txt", "w", encoding="utf-8") as f:
    f.write("comlpetion:" + completion)