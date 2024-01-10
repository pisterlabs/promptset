import os
from pathlib import Path
from tqdm import tqdm
import base64
import requests
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent
import pandas as pd
import openai
import multiprocessing as mp
mp.set_start_method('fork', force=True)
from catshand.utility import loggergen
from catshand.openai import process_audio_file, merge_tran_csv, convert_csv_to_txt, openai_text

def prjsummary(args):

    prjdir = Path(args.prj_dir)
    
    logdir = prjdir.joinpath('log')
    logdir.mkdir(exist_ok=True, parents=True)
    logger = loggergen(logdir)
    logger.info(f'args: {args}')

    threads = args.threads

    # check if output_dir specified
    if not args.input_dir is None:
        ipdir = Path(args.input_dir)
    else:
        ipdir = prjdir.joinpath('00_Raw_wav_processed_sil_removal')

    if not args.output_dir is None:
        opdir = Path(args.output_dir)
    else:
        opdir = prjdir.joinpath('transcription')

    csvdir = opdir.joinpath('csv')
    segdir = opdir.joinpath('wav')
    tmpdir = opdir.joinpath('tmp')
    docdir = opdir.joinpath('doc')
    txtdir = opdir.joinpath('txt')
    opdir.mkdir(exist_ok=True, parents=True)
    csvdir.mkdir(exist_ok=True, parents=True)
    docdir.mkdir(exist_ok=True, parents=True)
    txtdir.mkdir(exist_ok=True, parents=True)

    names = []

    audio_lsit = sorted(ipdir.glob('*.wav'))

    print(f'nCPU: {threads}')
    
    for ipfile in tqdm(audio_lsit):
        opfile = csvdir.joinpath(ipfile.relative_to(ipdir)).with_suffix('.csv')
        names.append(ipfile.stem)
        opsegdir = segdir.joinpath(ipfile.stem)
        opsegdir.mkdir(exist_ok=True, parents=True)
        logger.info(f'Processing Transcribe, save csv to : {opfile}')
        logger.info(f'Processing Transcribe, save wav files to : {opsegdir}')
        process_audio_file(ipfile, opfile, opsegdir, tmpdir, threads = threads)
    
    print(names)
    logger.info(f'merge csv files to: {docdir}')
    merge_tran_csv(csvdir, docdir)
    convert_csv_to_txt(docdir, txtdir)
    openai_text(docdir.joinpath('merge.txt'), docdir.joinpath('summary.txt'), names = names, threads = threads)
    
    return

def add_subparser(subparsers):
    description = "prjsummary creates the prejoct summary with transcript and time stamps"
    # parser = argparse.ArgumentParser(description=description)
    subparsers = subparsers.add_parser('prjsummary', help=description)
    required_group = subparsers.add_argument_group('Required Arguments')
    required_group.add_argument('-p', '--prj_dir', type = str, required = True, help = 'directory for the project folder')
    optional_group = subparsers.add_argument_group('Optional Arguments')
    optional_group.add_argument('-i', '--input_dir', type = str, help = 'input folders with *.wav files. Default folder: 00_Raw_wav_processed_sil_removal')
    optional_group.add_argument('-o', '--output_dir', type = str, help = 'output folders different from default')
    optional_group.add_argument('-t', '--threads', dest='threads', type=int, default = 1)
    subparsers.set_defaults(func=prjsummary)

    return