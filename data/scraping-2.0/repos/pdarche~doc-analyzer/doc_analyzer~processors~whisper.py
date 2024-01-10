import os
import openai
import time


class WhisperProcessor(object):
    def __init__(self, datadir, source_path, bucket=None):
        # Processors like textract will sometimes have 
        # some kind of python object intermediate representation
        # of the document.
        self.datadir = datadir
        self.source_path = source_path
        self.bucket = bucket

        self.num_subdocs = None
        self.num_pickles = None

    def process(self):
        return
    
    def get_text(self):
        # Get the audio path chunks
        chunks_path = f'{self.datadir}/audio/chunks'
        file_paths = [os.path.join(chunks_path, filename) for filename in sorted(os.listdir(chunks_path)) if filename.startswith('chunk')]
        
        # Create the outfult folder
        text_path = f'{self.datadir}/text/chunks'
        if not os.path.exists(text_path):
            os.mkdir(text_path)
        
        chunk_text = {} 
        for ix, file_path in enumerate(file_paths):
            chunk_text_path = f'{text_path}/chunk_{ix}.txt'
            if not os.path.exists(chunk_text_path):
                print(f"Chunk {ix} not found. Transcribing chunk: {file_path}")
                audio_file_chunk = open(file_path, 'rb')
                res = openai.Audio.transcribe(
                    'whisper-1',
                    audio_file_chunk 
                )
                text = res['text']
                chunk_text[ix+1] = text
                with open(chunk_text_path, 'w') as f:
                    f.write(text)
                time.sleep(1)
            else:
                print(f"Chunk {ix} found. Skipping transcription")
                with open(chunk_text_path, 'r') as f:
                    text = f.read()
                    chunk_text[ix+1] = text

        # Join the chunks together and 
        full_text = "\n".join(chunk_text.values())
        # Write the text file of the transcript
        # with open(f'{self.datadir}/text/{self.name}.txt', 'w') as f:
        #     f.write(self.text)
        return full_text, chunk_text
