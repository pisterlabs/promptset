import argparse
from Transcriber import Transcriber
from glob import glob

parser = argparse.ArgumentParser(description="Transcribe YouTube Videos Using Whisper API from OpenAI",
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-link", type=str,
                    help="link to YouTube video")
parser.add_argument("-gdrive", type=str, default=None,
                    help="path to google drive video")
parser.add_argument("-name", type=str,
                    help="name of the output")
parser.add_argument("-apikey", type=str, default=False,
                    help="OpenAI API key")


def console_entry():
    args = parser.parse_args()
    # Initialize transcriber
    transcriber = Transcriber(openai_key=args.apikey, model_size=None)
    # Download and save audio file
    if args.gdrive:
        transcriber.extract_audio_gdrive(args.gdrive)
    else:
        transcriber.download_audio(args.link)
    # Start transcribing
    for ind, audio_file in enumerate(sorted(glob("./Data/Chunks/*.m4a"))):
        result = transcriber.transcribe_api(audio_file)
        transcriber.write_api_result(result, args.name, ind)
    print("Transcribe completed!!")
    # Clear download folders
    transcriber.clear()


if __name__ == '__main__':
    console_entry()
