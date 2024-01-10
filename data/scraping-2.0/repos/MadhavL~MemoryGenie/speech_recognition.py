import pyaudio
import argparse
import asyncio
import json
import sys
import websockets
from langchain.text_splitter import RecursiveCharacterTextSplitter
from update_vector_db import update_db
import time
from query_vector_db import query_db_with_query

from datetime import datetime

startTime = datetime.now()

all_mic_data = []
all_transcripts = []

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 8000
KEY = "55c148996afae2c820da7699bd598492016434d2"

audio_queue = asyncio.Queue()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 115,
    chunk_overlap  = 0,
    length_function = len,
    add_start_index = True,
    separators=["\n\n", "\n", " ", ".", "?", "!"]
)

# Used for microphone streaming
def mic_callback(input_data, frame_count, time_info, status_flag):
    audio_queue.put_nowait(input_data)
    return (input_data, pyaudio.paContinue)


async def run(key):
    deepgram_url = 'wss://api.deepgram.com/v1/listen?punctuate=true&interim_results=true&model=nova-2&encoding=linear16&sample_rate=16000&endpointing=150'

    # Connect to the real-time streaming endpoint, attaching our credentials.
    async with websockets.connect(
        deepgram_url, extra_headers={"Authorization": f"Token {key}"}
    ) as ws:
        print("Opened Deepgram streaming connection")

        async def sender(ws):
            print("Ready to stream mic audio to Deepgram.")

            try:
                while True:
                    mic_data = await audio_queue.get()
                    all_mic_data.append(mic_data)
                    await ws.send(mic_data)
            except websockets.exceptions.ConnectionClosedOK:
                await ws.send(json.dumps({"type": "CloseStream"}))
                print("Closed Deepgram connection, waiting for final transcripts if necessary")

            except Exception as e:
                print(f"Error while sending: {str(e)}")
                raise

            return

        async def receiver(ws):
            """Print out the messages received from the server."""
            transcript = ""
            prev = ""
            prev_time = 0

            async for msg in ws:
                res = json.loads(msg)
                try:
                    # handle local server messages
                    if res.get("msg"):
                        print(res["msg"])
                    alternative = (res.get("channel", {}).get("alternatives", [{}])[0])
                    transcript = alternative.get("transcript", "")
                    if transcript != "":
                        #3 Conditions: 1) > 3s since last query, 2) > 25 characters, 3) > x characters of change
                        if time.time() - prev_time > 3 and len(transcript) > 25:
                            query = prev + transcript
                            prev_time = time.time()
                            # print(f"\nQUERY: {query}\n")
                            query_db_with_query(query)
                        
                        # print(f"{transcript}")
                        if res.get("is_final"):
                            if res.get("speech_final"):
                                # print("FINAL!")
                                prev = ""
                                if (transcript[-1] not in '.?!'):
                                    transcript += '.'
                            else:
                                prev += transcript + " "
                            all_transcripts.append(transcript)

                    # close stream if user says "goodbye"
                    if "goodbye" in transcript.lower():
                        if res.get("speech_final"):
                            del all_transcripts[-1]

                            #all_transcripts is the final transcription
                            conv_id = f"{int(time.time())}"

                            final_transcription = '\n'.join(all_transcripts)
                            # update_db([final_transcription], "full", conv_id)
                            chunks=[chunk.page_content for chunk in text_splitter.create_documents([final_transcription])]
                            print('\n')
                            for chunk in chunks:
                                print(chunk)
                                print('\n')

                            #chunks is the text we want to embed
                            # update_db(chunks, "sentence", conv_id)
                            
                        await ws.send(json.dumps({"type": "CloseStream"}))
                        print("Closed Deepgram connection, waiting for final transcripts if necessary")
                        

                except KeyError:
                    print(f"🔴 ERROR: Received unexpected API response! {msg}")

        # Set up microphone if streaming from mic
        async def microphone():
            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=mic_callback,
            )

            stream.start_stream()

            global SAMPLE_SIZE
            SAMPLE_SIZE = audio.get_sample_size(FORMAT)

            while stream.is_active():
                await asyncio.sleep(0.1)

            stream.stop_stream()
            stream.close()

        functions = [
            asyncio.ensure_future(sender(ws)),
            asyncio.ensure_future(receiver(ws)),
        ]

        
        functions.append(asyncio.ensure_future(microphone()))

        await asyncio.gather(*functions)


def main():
    """Entrypoint for the example."""
    try:
        asyncio.run(run(KEY))

    except websockets.exceptions.InvalidStatusCode as e:
        print(f'🔴 ERROR: Could not connect to Deepgram! {e.headers.get("dg-error")}')
        return
    except websockets.exceptions.ConnectionClosedError as e:
        error_description = f"Unknown websocket error."
        print(
            f"🔴 ERROR: Deepgram connection unexpectedly closed with code {e.code} and payload {e.reason}"
        )
        if e.reason == "DATA-0000":
            error_description = "The payload cannot be decoded as audio. It is either not audio data or is a codec unsupported by Deepgram."
        elif e.reason == "NET-0000":
            error_description = "The service has not transmitted a Text frame to the client within the timeout window. This may indicate an issue internally in Deepgram's systems or could be due to Deepgram not receiving enough audio data to transcribe a frame."
        elif e.reason == "NET-0001":
            error_description = "The service has not received a Binary frame from the client within the timeout window. This may indicate an internal issue in Deepgram's systems, the client's systems, or the network connecting them."

        print(f"🔴 {error_description}")
        return

    except websockets.exceptions.ConnectionClosedOK:
        return

    except Exception as e:
        print(f"🔴 ERROR: Something went wrong! {e}")
        return


if __name__ == "__main__":
    sys.exit(main() or 0)
