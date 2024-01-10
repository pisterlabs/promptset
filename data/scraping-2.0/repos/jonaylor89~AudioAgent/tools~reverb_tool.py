
import os
import sys
import asyncio
import subprocess
from typing import Optional 
from langchain.callbacks.manager import (
    CallbackManagerForToolRun,
    AsyncCallbackManagerForToolRun,
)
from langchain.tools import BaseTool, StructuredTool

from pedalboard import Pedalboard, Chorus, Compressor, Delay, Gain, Reverb, Phaser
from pedalboard.io import AudioStream
from tqdm import tqdm
import soundfile as sf
import numpy as np

BUFFER_SIZE_SAMPLES = 1024 * 16
NOISE_FLOOR = 1e-4

def get_num_frames(f: sf.SoundFile) -> int:
    # On some platforms and formats, f.frames == -1L.
    # Check for this bug and work around it:
    if f.frames > 2 ** 32:
        f.seek(0)
        last_position = f.tell()
        while True:
            # Seek through the file in chunks, returning
            # if the file pointer stops advancing.
            f.seek(1024 * 1024 * 1024, sf.SEEK_CUR)
            new_position = f.tell()
            if new_position == last_position:
                f.seek(0)
                return new_position
            else:
                last_position = new_position
    else:
        return f.frames


def reverb_file(input_filename: str):
    """adds reverb to an audio file and saves the output to a new file"""

    if not os.path.isfile(input_filename):
        return f"input file '{input_filename}' does not exist, the input must be a valid filename and exist"

    cut_reverb_tail = False
    output_file = input_filename.replace(".wav", "_reverb.wav")

    reverb = Reverb()
    with sf.SoundFile(input_filename) as input_filename:
        sys.stderr.write(f"\nWriting to {output_file}...\n")
        # if os.path.isfile(output_file) and not args.overwrite:
        #     raise ValueError(
        #         f"Output file {output_file} already exists! (Pass -y to overwrite.)"
        #     )
        with sf.SoundFile(
            output_file,
            'w',
            samplerate=input_filename.samplerate,
            channels=input_filename.channels,
        ) as output_file:
            length = get_num_frames(input_filename)
            length_seconds = length / input_filename.samplerate
            sys.stderr.write(f"Adding reverb to {length_seconds:.2f} seconds of audio...\n")
            for dry_chunk in input_filename.blocks(BUFFER_SIZE_SAMPLES, frames=length):
                # Actually call Pedalboard here:
                # (reset=False is necessary to allow the reverb tail to
                # continue from one chunk to the next.)
                effected_chunk = reverb.process(
                    dry_chunk, sample_rate=input_filename.samplerate, reset=False
                )
                # print(effected_chunk.shape, np.amax(np.abs(effected_chunk)))
                output_file.write(effected_chunk)

            if not cut_reverb_tail:
                while True:
                    # Pull audio from the effect until there's nothing left:
                    effected_chunk = reverb.process(
                        np.zeros((BUFFER_SIZE_SAMPLES, input_filename.channels), np.float32),
                        sample_rate=input_filename.samplerate,
                        reset=False,
                    )
                    if np.amax(np.abs(effected_chunk)) < NOISE_FLOOR:
                        break
                    output_file.write(effected_chunk)

    return f"completed reverb on {input_filename} and saved the output to {output_file}"


ReverbTool = StructuredTool.from_function(reverb_file)