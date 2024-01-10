import argparse
import json
import logging
import os

from dotenv import load_dotenv
import openai

from helper import sanitize_file_name
from podcast_diarization_to_tracks import do_diarization
from podcast_tracks_to_srts import transcription_with_backoff
from podcast_srts_to_transcript import create_script
from podcast_transcript_gpt_fixup import fixup_with_backoff


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='podcast_diarization_to_tracks',
                    description='Parses an MP3 file into separate speaker chunks'+
                                ' by diarization. Generates a lot of files in output/[input_file_name]/'+
                                ' including a file durations_and_speakers.csv',
                    epilog='Run podcast_tracks_to_srts.py after completed')
    parser.add_argument('--file', metavar='-f', required=True,
                    help='input file to process')
    parser.add_argument('--num_speakers', metavar='-s', type=int, default=None, required = False,
                    help='best guess on number of speakers in file, or don\'t specify')
    parser.add_argument('--names', metavar='-n', default="[]", required = False,
                    help='Names of speakers in json array format: ' + 
                          '[{\\"name\\":\\"Drew\\",\\"speaker_id\\":\\"SPEAKER_00\\"},{\\"name\\":\\"Kristin\\",\\"speaker_id\\":\\"SPEAKER_01\\"}]' )    
    parser.add_argument('--output', metavar='-o', default='output', required = False,
                    help='output directory to work in')
    parser.add_argument('-l', '--loglevel', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    default='INFO', help='Set the logging level')
    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel)
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_TOKEN')
    hf_key = os.getenv('HUGGING_FACE_TOKEN')

    print (openai.api_key)
    print (hf_key)

    logging.info("do_diarization")
    do_diarization(args.file, hf_key, args.output, args.num_speakers)
    
    logging.info("transcription_with_backoff")
    transcription_with_backoff(filename = sanitize_file_name(args.file), output_dir = args.output)
    logging.info(transcription_with_backoff.retry.statistics)
    
    logging.info("create_script")
    create_script(speakers = json.loads(args.names), 
                  output_path=args.output + '\\' + sanitize_file_name(args.file))
    
    logging.info("fixup_with_backoff")
    fixup_with_backoff(output_path = args.output + '\\' + sanitize_file_name(args.file))
    logging.info(fixup_with_backoff.retry.statistics)

    