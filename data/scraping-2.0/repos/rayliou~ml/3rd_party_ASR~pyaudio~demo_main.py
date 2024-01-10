#!/home/xiaorui/anaconda3/bin/python3
import asyncio
import sys,time
import subprocess
import numpy as np
import configparser
import argparse
import aiohttp
import struct
import logging
import websockets,json

from mic_device import MicDevice


def create_wave_header(sample_rate=16000, bits_per_sample=16, num_channels=1, num_frames=16000*120):
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    """ https://docs.fileformat.com/audio/wav/
    1 - 4	“RIFF”	Marks the file as a riff file. Characters are each 1 byte long.
    5 - 8	File size (integer)	Size of the overall file - 8 bytes, in bytes (32-bit integer). Typically, you’d fill this in after creation.
    9 -12	“WAVE”	File Type Header. For our purposes, it always equals “WAVE”.
    13-16	“fmt "	Format chunk marker. Includes trailing null
    17-20	16	Length of format data as listed above
    21-22	1	Type of format (1 is PCM) - 2 byte integer
    23-24	2	Number of Channels - 2 byte integer
    25-28	44100	Sample Rate - 32 byte integer. Common values are 44100 (CD), 48000 (DAT). Sample Rate = Number of Samples per second, or Hertz.
    29-32	176400	(Sample Rate * BitsPerSample * Channels) / 8.
    33-34	4	(BitsPerSample * Channels) / 8.1 - 8 bit mono2 - 8 bit stereo/16 bit mono4 - 16 bit stereo
    35-36	16	Bits per sample
    
    37-40	“data”	“data” chunk header. Marks the beginning of the data section.
    41-44	File size (data)	Size of the data section.
Sample values are given above for a 16-bit stereo source.
    """
    # https://docs.python.org/3/library/struct.html
    wave_header = struct.pack( '<4sI4s4sIHHIIHH4sI',
                              b'RIFF',  # RIFF format
                              36 + num_frames * block_align,  # ChunkSize
                              b'WAVE',  # 'WAVE' format
                              b'fmt ',  # 'fmt ' subchunk
                              16,  # Subchunk1Size
                              1,  # AudioFormat (PCM)
                              num_channels,  # NumChannels
                              sample_rate,  # SampleRate
                              byte_rate,  # ByteRate
                              block_align,  # BlockAlign
                              bits_per_sample,  # BitsPerSample
                              b'data',  # 'data' subchunk
                              num_frames * block_align  # Subchunk2Size
                              )

    return wave_header
class OpenAIGPTLLMClient:
    def __init__(self, config):
        self.logger = logging.getLogger("main." + self.__class__.__name__)
        self.api_key = config.get(self.__class__.__name__, 'api_key', fallback='')
        self.enable = config.get(self.__class__.__name__, 'enable', fallback=True)
        self.enable = False if self.enable.strip().lower() == "false" else True
        self.logger.debug(f"api_key:{self.api_key}")
        self.session = aiohttp.ClientSession()

    async def send_and_receive(self, question):
        """ https://platform.openai.com/docs/api-reference/making-requests """
        if not self.enable: 
            return "The LLM is disable"
        url = "https://api.openai.com/v1/chat/completions"
        headers =  {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        json_body  = {
            "temperature": 0.5,
            "model": "gpt-3.5-turbo",
             "messages": [
                 {"role": "system", "content": "You are a general AI assistant; please respond to questions based on my needs, using the language of the question"},
                 {"role": "user", "content": question}
             ],
        }
        async with self.session.request("POST", url, headers=headers, json=json_body) as response:
            ret =  await response.json()
            try:
                ret = ret['choices'][0]['message']['content']
            except Exception as e:
                self.logger.warning(f"An error occurred: {e}")
                ret = "An error occurred when fetching result from OpenAI."
            return ret

    async def close(self):
        await self.session.close()  # 关闭aiohttp的会话

class WsDeepgramASRClient:
    def __init__(self, config):
        self.logger = logging.getLogger("main." + self.__class__.__name__)
        self.api_key_ = config.get(self.__class__.__name__, 'DEEPGRAM_API_KEY', fallback='xxxxxxx')
        self.host_ = config.get(self.__class__.__name__, 'host', fallback="wss://api.deepgram.com")
        self.language_  = config.get(self.__class__.__name__, 'language', fallback="en")
        self.interim_results_ = config.get(self.__class__.__name__, 'interim_results', fallback="true")
        self.disconnected_ = False
        self.text_updated_ = False
        self.text_ = ""
        self.start_time_ = time.time()
        self.record_mode_ = config.getboolean(self.__class__.__name__, 'record_mode', fallback=False)
        if self.record_mode_:
            self.record_file_ = config.get(self.__class__.__name__, 'record_file', fallback='record.wav')
            self.file_ = None
        self.ws_ = None

    async def connect(self):
        if self.record_mode_:
            self.file_ = open(self.record_file_, 'wb')
        deepgram_url = f'{self.host_}/v1/listen?punctuate=true'
        deepgram_url += f"&interim_results={self.interim_results_}&language={self.language_}"
        deepgram_url += "&encoding=linear16&sample_rate=16000"
        extra_headers = {"Authorization": "Token {}".format(self.api_key_)}
        try:
            #self.logger.debug(f'Calling websockets.connect({deepgram_url}, extra_headers={extra_headers})')
            self.ws_ = await websockets.connect(deepgram_url, extra_headers=extra_headers)
            self.logger.info(f'ℹ️  Request ID: {self.ws_.response_headers.get("dg-request-id")}')
            self.logger.info("🟢 (1/5) Successfully opened Deepgram streaming connection")
            asyncio.create_task(self.receiver(self.ws_))
        except websockets.exceptions.InvalidStatusCode as e:
            self.logger.fatal(f"WebSocket connection failed with status code {e.status_code}. Response headers: {e.response_headers}")
            sys.exit(1)

    async def send_audio(self, audio):
        if self.disconnected_:
            self.logger.debug(f"disconnected_:{self.disconnected_};ws:{self.ws_}")
            return
        if self.record_mode_:
            self.file_.write(audio)
        try:
            await self.ws_.send(audio)
        except websockets.exceptions.ConnectionClosedOK:
            #await self.ws_.send(json.dumps({"type": "CloseStream"}))
            self.logger.warning( "🟢 (5/5) Successfully closed Deepgram connection, waiting for final transcripts if necessary")
            self.disconnected_ = True
            pass
        except Exception as e:
            self.logger.error (f"Error while sending: {str(e)}")
            self.disconnected_ = True
            raise

    async def receiver(self, ws):
        first_message = True
        async for msg in ws:
            res = json.loads(msg)
            if first_message:
                self.logger.info(
                    "🟢 (3/5) Successfully receiving Deepgram messages, waiting for finalized transcription..."
                )
                first_message = False
            try:
                transcript = (
                    res.get("channel", {})
                    .get("alternatives", [{}])[0]
                    .get("transcript", "")
                ).strip()
                if res.get("is_final"):
                    self.text_ += transcript
                if transcript != "":
                    self.text_updated_ = True
                    self.logger.debug(transcript)
                if res.get("created"):
                    self.logger.info(
                        f'🟢 Request finished with a duration of {res["duration"]} seconds. Exiting!'
                    )
            except KeyError:
                print(f"🔴 ERROR: Received unexpected API response! {msg}")

                #if method == "mic" and "goodbye" in transcript.lower():
                #    await ws.send(json.dumps({"type": "CloseStream"}))
                #    print(
                #        "🟢 (5/5) Successfully closed Deepgram connection, waiting for final transcripts if necessary"
                #    )
    def handle_close(self, code):
        self.disconnected_ = True
        self.logger.warning(f'Connection closed with code {code}.')


    async def disconnect(self):
        if self.record_mode_:
            self.file_.close()
            self.logger.info("Close record audio file")
        if self.ws_:
            self.logger.info("Close deepgramLive_")
            self.disconnected_ = True
            await self.ws_.send(json.dumps({"type": "CloseStream"}))
            await self.ws_.close()


class DeviceManager:
    def __init__(self,config, device, asr_client, llm_client):
        self.asr_session_timeout_ = float(config.get(self.__class__.__name__, 'asr_session_timeout', fallback=20))
        self.non_speech_timeout_  = float(config.get(self.__class__.__name__, 'non_speech_timeout', fallback=2.5))
        self.logger = logging.getLogger("main." + self.__class__.__name__)
        self.device_ = device
        self.asr_client_ = asr_client
        self.llm_client_ = llm_client


    async def disconnect_asr_after_timeout(self, timeout):
        secs_passed = 0
        non_speech_secs_passed  = 0
        wait_1st_reply = True
        while secs_passed < timeout:
            if int(2 * secs_passed) % 2 == 0:
                #self.logger.debug("=" * int(secs_passed))
                pass
            await asyncio.sleep(0.5)
            secs_passed += 0.5
            non_speech_secs_passed += 0.5
            non_speech_timeout =  4.5 if wait_1st_reply else self.non_speech_timeout_
            if self.asr_client_.text_updated_:
                self.logger.warning("->" * int(secs_passed))
                non_speech_secs_passed = 0.0
                self.asr_client_.text_updated_ = False
                wait_1st_reply = False
            elif non_speech_secs_passed > non_speech_timeout:
                self.logger.debug(f"No more transcription after {non_speech_secs_passed}s")
                break

        await self.asr_client_.disconnect()
        self.logger.debug(f"Disconnect the asr due to the timeout {timeout} s")

    async def connect_asr(self):
        await self.asr_client_.connect()

    async def play_audio(self):
        await asyncio.sleep(3)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.run_shell_command)

    def run_shell_command(self):
        subprocess.Popen(["aplay", "./10secs_english_speech.wav"])

    async def start(self):
        await self.connect_asr()
        #audio_data = create_wave_header()
        #await self.asr_client_.send_audio(audio_data)
        #play_audio_task = asyncio.create_task(self.play_audio())
        timeout = self.asr_session_timeout_
        self.logger.debug(f"create a timeout task for  {timeout} s" )
        disconnect_task = asyncio.create_task(self.disconnect_asr_after_timeout(timeout))
        try:
            cnt = 0
            while not self.asr_client_.disconnected_:
                audio_data = await self.device_.get_data()
                if audio_data is None:
                    break
                await self.asr_client_.send_audio(audio_data)
            question = self.asr_client_.text_.strip()
            self.logger.warning(f"Question from ASR: [{question}]")
            answer = await self.llm_client_.send_and_receive(question)
            self.logger.warning(f"Answer from LLM: [{answer}]")
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
        self.device_.cleanup()

async def main():
    parser = argparse.ArgumentParser(description='My Script')
    parser.add_argument('--log_level', choices=['debug', 'info', 'warning', 'error', 'critical'],
                        default=None, help='Set the log level from command line.')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read('config.ini')
    if args.log_level:
        log_level = getattr(logging, args.log_level.upper())
    else:
        log_level = config.get('Logging', 'log_level', fallback='DEBUG')
        log_level = getattr(logging, log_level.upper())
    logger = logging.getLogger('main')
    logger.setLevel(log_level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    def getch():
        try:
            # Unix-like systems
            import termios, tty
            old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setcbreak(sys.stdin.fileno())
                char = sys.stdin.read(1)
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        except ImportError:
            # Windows systems
            import msvcrt
            char = msvcrt.getch().decode("utf-8")
        return char


    while True:
        print("Press 'q' to quit.")
        print("Press 'g' or ' ' (space) to enter Cloud-based ASR-LLM.")
        user_input = getch().lower()
        if user_input == 'q':
            print("Exiting the event loop. Goodbye!")
            break
        elif user_input in ['g', ' ']:
            print("Entering Cloud-based ASR-LLM.")
            device = MicDevice(config)
            asr_client = WsDeepgramASRClient(config)
            llm_client = OpenAIGPTLLMClient(config)
            device_manager = DeviceManager(config, device, asr_client, llm_client)
            await device_manager.start()
        else:
            print(f"Invalid input [{user_input}] . Please try again.")


if __name__ == "__main__":
    asyncio.run(main())
