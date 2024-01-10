import whisper
import openai
import os 
from dotenv import load_dotenv

# This function loads environment variables from a .env file into the script's environment.
load_dotenv() 

# The OpenAI API key is retrieved from an environment variable OPENAI_API_KEY which should be defined in your .env file. This is a secure way to handle credentials.
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load the whisper model
model = whisper.load_model("base")

# Transcribe the audio file. You must provide the absolute path to the audio file.
result = model.transcribe("C:/Users/user/audio/output_audio.mp3")

# This is a utility function to write text to a specified file. It opens a file at the given file_path in write mode, encodes the content in UTF-8, and writes the text into it.
def write_text_to_file(text, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)

# This function generates a text summary using the OpenAI GPT model specified by the model parameter. It takes the text to summarize, the model to use, and the maximum number of tokens for the summary as parameters.
# The summary prompt includes instructions for generating a concise summary and bullet points for the next steps and action items.
# The temperature is set to 0.2 to encourage the model to produce more deterministic outputs, reducing randomness.
def get_summary(text, model="gpt-4-1106-preview", max_tokens=150):
    """
    Generate a summary using OpenAI's GPT model.

    :param text: str, the text to summarize
    :param model: str, the model to use for summarization
    :param max_tokens: int, the maximum length of the summary
    :return: str, the generated summary
    """
    response = openai.Completion.create(
        engine=model,
        prompt=f"Please generate a concise summary of the key topics discussed in the meeting, including the main ideas and decisions made. Additionally, provide bullet points listing the next steps and action items that should be taken following the meeting. Ensure the summary is clear, coherent, and ready to be shared with participants for review and follow-up. Meeting Transcrip:\n{text}\n",
        max_tokens=max_tokens,
        temperature=0.2
    )
    
    summary = response.choices[0].text.strip()
    return summary

# The script takes the transcribed text from the result and assigns it to summary.
summary = result["text"]

# The script uses the write_text_to_file function to save the summary to a file located at "summaries\summaryT0.2Mgpt-3.5-turbo-instruct-0914.txt".
write_text_to_file(summary, "summaries\summaryT0.2Mgpt-3.5-turbo-instruct-0914.txt")
