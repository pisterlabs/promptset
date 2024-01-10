import sys
from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    # api_key="My API Key",
)
from docx import Document

def transcribe_audio(audio_file_path):

    with open(audio_file_path, 'rb') as audio_file:
        transcription = client.audio.transcriptions.create(model="whisper-1", file=audio_file, prompt="Interview of Alex for Senior Engineering Leadership role. Finix, Rocket Lawyer, Bigfoot Biomedical, Abbott, GE, Google, WebLogic, Division.")
    return transcription.text

def save_text_to_file(text, filesname):
    with open(filesname, "w") as f:
        f.write(text)

def analyze_interview(text):
    chatCompletion = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a Career Interview Coach highly skilled in language comprehension and analysis of technical interview techniques. Please read the following transcription of a technical interview for a Senior Engineering Leadership role and identify each question asked by the interviewer, then provide coaching guidance and feedback on the answer provided by the interviewee. Focus the most important points, providing a coherent and readable summary that could help the interviewee improve their interview technique to increase the liklihood of recieving a job offer. Please avoid unnecessary details or tangential points."
            },
            {
                "role": "user",
                "content": text
            }
        ]
    )
    print("Chat Completion = \n" + str(vars(chatCompletion)))
    return chatCompletion.choices[0].message.content

def main(argv):    

  if len(sys.argv) < 2:
        print("Usage: python voice2text.py [-o Sentiment] <input_file>")
        sys.exit(1)

  # Read on options flag from the command line to perform different actions
  options_flag = argv[1]
  option_command = "transcribe"

  if options_flag == "-o":
      option_command = argv[2]
      input_file = argv[3]

  # if option_command == "analyze":
  #   # Get the sentiment of a text snippet
  #   # read in the content of the text file
  #   with open(input_file, 'r') as file:
  #       text = file.read()
  #   sentiment = analyze_interview(text)
  #   save_text_to_file(sentiment, input_file.split('.')[0] + ' (analysis).txt')
  # elif option_command == "transcribe":
    
  transcription = transcribe_audio(input_file)
  # Save the result to a text file in the current directory with the same name as the audio file
  save_text_to_file(transcription, input_file.split('.')[0] + ' (transcription).txt')


# Run the main function if invoked as a script
if __name__ == "__main__":
    main(sys.argv)
    