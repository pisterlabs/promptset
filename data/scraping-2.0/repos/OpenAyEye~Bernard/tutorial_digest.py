import os
import yt_dlp
import speech_recognition as sr
from dotenv import load_dotenv
import openai
import whisper
import json
from bs4 import BeautifulSoup
import requests
import time




model = whisper.load_model("medium", device="cuda") #Change to "cpu" for CPU

functions = [
    {
        "name": "digest_content",
        "description": "generates detailed hierarchical outlines of content including clarifying notes and commentary.",
        "parameters": {
            "type": "object",
            "properties": {
                "digested_content": {
                    "type": "string",
                    "description": "A comprehensive outline and summary of the content"
                }
            }
        }
    },
{
        "name": "summarize_content",
        "description": "digest content and write verbose detailed lectures on the content provided",
        "parameters": {
            "type": "object",
            "properties": {
                "content_summary": {
                    "type": "string",
                    "description": "A comprehensive, detailed lecture and summary of the content"
                }
            }
        }
    },
]


# Load environment variables from the .env file
load_dotenv("config.env")

# Access the OpenAI key from the environment variable
openai.api_key = os.environ.get("OpenAiKey")
# bing_u_cookie = os.environ.get("bing_u_cookie")
def download_audio(youtube_url, output_dir):
    print("Ripping Audio")
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_dir, '%(title)s'),  # Specify the filename with .wav extension
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(youtube_url, download=True)
        audio_file_path = ydl.prepare_filename(result)
        video_title = result.get('title', 'untitled')  # Get the title of the video, use 'untitled' as a fallback
    # Return the path of the downloaded audio file
    return audio_file_path + ".wav", video_title

def transcribe_audio(AudioFile, text_file_path):

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(AudioFile)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions(fp16=True) # Change to False for CPU. check line 89 for further changes that need to be made for CPU use.
    result = model.transcribe(AudioFile)
    print(result["text"])


    transcription = result["text"]

    with open(text_file_path, "w", encoding="utf-8") as text_file:
        text_file.write(transcription)
    #print(transcription)
    return transcription

def content_summary(user_input, outline):
    # Removing the function call here opens GPT up a little, using function calling resulted in 80-400
    # token responses. instead of the expected 10k token responses.
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=[
            {
                "role": "system",
                "content": "you are a useful essayist. You take content and develop comprehensive lectures to explain "
                           "the content explainer."

            },
            {
                "role": "user",
                "content": f"write a long form, comprehensive and detailed 2000 word essay for senior year college : "
                           f"students discussing '{user_input}'." # the lecture needs to cover the following "
                           #f"outline point for point: '{outline}'"


            }
        ],
        #functions=functions,
        #function_call={
        #    "name": functions[1]["name"]
        #},
        max_tokens=10000
    )
    time.sleep(1)
    bot_response = response["choices"][0]["message"]["content"]
    return bot_response
    #arguments = response["choices"][0]["message"]["function_call"]["arguments"]
    #json_obj = json.loads(arguments)
    #print(f"content summary: {json_obj['content_summary']}")
    #return str(json_obj["content_summary"])
def digest_content(user_input):
    print("Digesting Content!")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=[
            {
                "role": "system",
                "content": "You are a useful Outline notation note taking assistant. You're an expert in providing "
                           "comprehensive notes using the hierarchical outlines with lots of detail and clarifying "
                           "commentary."
            },
            {
                "role": "user",
                "content": f"write college level notes for the following content, use The standard hierarchical "
                           f"outline format for note-taking commonly known as 'Outline notation' or 'Outline "
                           f"format. ie: '1., 1.a, 1.b, 2. 2.a,2.b, etc' Go several levels deep, make sure and note the details, include notes of "
                           f"important key aspects of the content: {user_input}."

            }
        ],
        functions=functions,
        function_call={
            "name": functions[0]["name"]
        }
    )
    print("content digested.")

    try:
        arguments = response["choices"][0]["message"]["function_call"]["arguments"]
        json_obj = json.loads(arguments)
        return json_obj["digested_content"]
    except json.decoder.JSONDecodeError as e:
        # Handle the JSONDecodeError gracefully
        print(f"Error while decoding JSON data: {e}")
        return None


def scrape_webpage_content(webpage_url):
    response = requests.get(webpage_url)
    if response.status_code != 200:
        print(f"Failed to fetch the webpage: {response.status_code}")
        return None

    soup = BeautifulSoup(response.content, "html.parser")

    main_content_tags = ["p", "h1", "h2", "h3", "h4", "h5", "h6"]

    # Additional filtering criteria
    unwanted_classes = ["menu", "advertisement", "sidebar", "footer", "a", "li"]
    unwanted_tags = ["nav", "aside", "footer", "header", "span", "li class"]
    min_content_length = 50
    max_position_to_include = 5

    clean_text = ""
    for i, tag in enumerate(soup.find_all(main_content_tags)):
        # Exclude elements with certain class names
        if tag.get("class") and any(cls in tag.get("class") for cls in unwanted_classes):
            tag.extract()  # Remove the unwanted tag
            continue

        # Exclude specific tags
        if tag.name in unwanted_tags:
            tag.extract()  # Remove the unwanted tag
            continue

        text = tag.get_text().strip()

        # Exclude elements with content length below the threshold
        if len(text) < min_content_length:
            continue

        # Exclude elements within certain positions in the document
        if i < max_position_to_include:
            continue

        clean_text += f"{text}\n"

    return clean_text

def split_text_into_chunks(text, max_tokens):
    chunks = []
    words = text.split()
    current_chunk = ""

    for word in words:
        if len(current_chunk) + len(word) + 1 <= max_tokens:
            current_chunk += " " + word if current_chunk else word
        else:
            chunks.append(current_chunk)
            current_chunk = word

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def main():
    input_url = input("Enter URL: ")
    #input_url = "youtube"
    output_dir = "audio_files"  # Change this directory path if you want a different output location
    digested_content = "none"
    summarized_content = "none"
    content_return = "none"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if "youtube" in input_url:
        #audio_file_path = download_audio(input_url, output_dir)
        audio_file_path, video_title = download_audio(input_url, output_dir)
        ######################################TESTING###############################################################################
        #audio_file_path= "audio_files/ChatGPT as an Interpreter： Introducing the KB Microservice for autonomous AI entities.txt"
        #with open(audio_file_path, "r") as file:
        #    file_content = file.read()
        #text = file_content
        #print(audio_file_path)
        ############################################END TESTING####################################################################
        text_file_path = os.path.join(os.path.splitext(audio_file_path)[0] + ".txt")
        text = transcribe_audio(audio_file_path, text_file_path)
        if text is not None:

            summary_text_file_path = os.path.join(os.path.splitext(audio_file_path)[0] + f"_summary.txt")


            print("Transcription: ")
            print(text)
            digested_content = digest_content(text)
            summarized_content = content_summary(text, digested_content)
            content_return = f"Outline:\n {digested_content} \nSummary: \n{summarized_content}"



            page_name = os.path.basename(input_url).split(".")[0]
            # Create the 'digested' directory if it doesn't exist
            if not os.path.exists("digested"):
                os.makedirs("digested")

            # Save the digested content to a text file
            output_file_path = os.path.join("digested", f"{video_title}_summary.txt")
            with open(output_file_path, "w", encoding="utf-8") as text_file:
                text_file.write(content_return)

            #with open(summary_text_file_path, "w", encoding="utf-8") as text_file:
            #    text_file.write(summarized_content)
        else:
            print(f"Transcription for {audio_file_path} was unsuccessful.")
        ################################TESTING#############################################
        #print("Summary and Outline: ")
        #print("**********************")
        #print(f"Summary: {summarized_content}")
        #print(f"Outline: {digested_content}")
        #############################END TESTING##############################################

        print(content_return)
        return content_return
    else:
        print("not a youtube channel, will scrape for text and digest")
        webpage_url = input_url  # Assuming the input is a webpage URL
        webpage_content = scrape_webpage_content(webpage_url)
        digested_content = digest_content(webpage_content)
        summarized_content = content_summary(webpage_content, digested_content)
        #########################TESTING###############################
        #print("Summary and Outline: ")
        #print("**********************")
        #print(f"Summary: \n{summarized_content}")
        #print(f"Outline: \n{digested_content}")
        ######################END TESTING##############################
        # Save the digested content to a text file if needed
        content_return = f"Outline:\n {digested_content} \nSummary: \n{summarized_content}"

        # Extract the page name from the URL
        page_name = os.path.basename(webpage_url).split(".")[0]

        # Create the 'digested' directory if it doesn't exist
        if not os.path.exists("digested"):
            os.makedirs("digested")

        # Save the digested content to a text file
        output_file_path = os.path.join("digested", f"{page_name}_summary.txt")
        with open(output_file_path, "w", encoding="utf-8") as text_file:
            text_file.write(content_return)
        #with open("webpage_content.txt", "w", encoding="utf-8") as text_file:
        #    text_file.write(webpage_content)
        #with open("digested_content.txt", "w", encoding="utf-8") as text_file:
        #    text_file.write(digested_content + " " + summarized_content)


        print(content_return)
        return content_return

if __name__ == "__main__":
    main()
