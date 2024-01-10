import logging.config
import openai

from scripts.utils import parse_config, get_output_file_path, arg_parser

logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def audio_to_text(audio_file_path: str, transcript_file_path: str):
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
    except Exception as e:
        logging.error(f"An error occurred during the transcribe process: '{e}'")

    try:
        with open(transcript_file_path, "w") as transcript_file:
            transcript_file.write(transcript["text"])
    except Exception as e:
        logging.error(f"An error occurred during saving transcripts: '{e}'")


def main():
    # get config data
    config = parse_config("config.ini")
    openai.api_key = config.get("api_keys", "OPENAI_KEY")

    args = arg_parser()

    # prepare the output path
    transcript_file_path = get_output_file_path(args.input_file_path, args.output_file_path)

    audio_to_text(args.input_file_path, transcript_file_path)
    logging.info(f"The transcript has been saved to the file - {transcript_file_path}")


if __name__ == "__main__":
    main()
