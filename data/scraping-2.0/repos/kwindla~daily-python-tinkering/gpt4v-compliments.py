#
#
#
# Usage: python3 deepgram_tts.py -m <daily_url> -t "<prompt>"
#
#

import argparse
import threading
import requests
import time
import io
import os
import base64

from openai import OpenAI
from PIL import Image
from daily import *

SAMPLE_RATE = 24000
NUM_CHANNELS = 1


class AudioBuffer:
    def __init__(self):
        self.buffer = io.BytesIO()
        self.lock = threading.Lock()

        # False as long as we have not received the last chunk of audio
        self.finished = False

        self.bytes_per_100ms = int(SAMPLE_RATE * 2 * NUM_CHANNELS * 0.1)

    def add_data(self, data):
        with self.lock:
            self.buffer.write(data)

    def get_data(self):
        with self.lock:
            # return None if we don't have 100ms of audio and we aren't finished
            available = self.buffer.tell()
            if available < self.bytes_per_100ms and not self.finished:
                return None

            size = min(available, self.bytes_per_100ms)
            self.buffer.seek(0)
            data = self.buffer.read(size)
            remaining = self.buffer.read()
            self.buffer = io.BytesIO(remaining)
            self.buffer.seek(0, 2)  # move to the end for next write
            return data


class AudioApp:
    def __init__(self):
        self.__virtual_mic = Daily.create_microphone_device(
            "my-mic",
            sample_rate=SAMPLE_RATE,
            channels=NUM_CHANNELS
        )

        self.__client = CallClient()

        self.__client.update_inputs({
            "camera": False,
            "microphone": {
                "isEnabled": True,
                "settings": {
                    "deviceId": "my-mic",
                }
            }
        }, completion=self.on_inputs_updated)

        self.__client.update_subscription_profiles({
            "base": {
                "camera": "unsubscribed",
                "microphone": "unsubscribed"
            }
        })

        self.__app_quit = False
        self.__app_error = None
        self.__app_joined = False
        self.__app_inputs_updated = False

        self.__start_event = threading.Event()
        self.__send_thread = threading.Thread(target=self.send_raw_audio)
        self.__send_thread.start()

        self.__text = None

        self.__seen_uuids = set()

    def on_inputs_updated(self, inputs, error):
        if error:
            print(f"Unable to updated inputs: {error}")
            self.__app_error = error
        else:
            self.__app_inputs_updated = True
        self.maybe_start()

    def on_joined(self, data, error):
        if error:
            print(f"Unable to join meeting: {error}")
            self.__app_error = error
        else:
            self.__app_joined = True
        self.maybe_start()

    def frame_to_jpg(self, frame):
        img = Image.frombuffer('RGB', (frame.width, frame.height),
                               frame.buffer, 'raw', 'RGB', 0, 1)
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        return buf.getvalue()

    def on_video_frame(self, participant_id, video_frame):
        print(f"on_video_frame: {participant_id}")
        uuid = participant_id.strip('"')  # bug? report to Aleix?
        self.__client.update_subscriptions({uuid: {'media': {
            "camera": "unsubscribed",
            "microphone": "unsubscribed"}
        }})

        # unsubscribe isn't always working? track this down later
        if uuid in self.__seen_uuids:
            return
        self.__seen_uuids.add(uuid)

        jpg = self.frame_to_jpg(video_frame)
        b64jpg = base64.b64encode(jpg).decode("utf-8")

        print("have frame, will travel")

        participants = self.__client.participants()

        identityPromptFragment = ""
        if participants[uuid]['info'].get('userName', None):
            identityPromptFragment = f"""
            The person in the image has identified themself as 
            {participants[uuid]['info']['userName']}. Please use their name
            when you compliment them if you can.
            """

        prompt = f"""
        You are the emcee at a holiday party. Here is an image from the party.
        If a person is in the image, give the person a nice compliment about what
        they are wearing. {identityPromptFragment}
        If the image is blank, just respond "pass".
        If there is no person in the image, describe the image briefly.
        Begin your response by saying "Starting."
        """

        gpt = OpenAI()
        response = gpt.chat.completions.create(
            model="gpt-4-vision-preview",
            max_tokens=250,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64jpg}"
                            }
                        }
                    ]
                }
            ]
        )
        print(response)
        message = response.choices[0].message.content
        print(message)
        self.fetch_audio(message)
        print("DONE WITH FRAME FROM", uuid)

    def run(self, meeting_url, text):
        print(f"running ... {meeting_url}")
        self.__text = text

        self.__client.join(meeting_url, completion=self.on_joined)

        self.__raw_audio_buffer = AudioBuffer()
        # self.__fetch_thread = threading.Thread(target=self.fetch_audio)
        # self.__fetch_thread.start()

        # loop through participants, grabbing a frame, sending the frame
        # to gpt4v, and sending the response into the session
        time.sleep(2)
        participants = self.__client.participants()
        for uuid, participant in participants.items():
            if participant['info']['isLocal']:
                continue
            print(f"uuid: {uuid}")
            print(f"participant: {participant}")
            # print(
            #    f"participant: {participant['info'].get('userName', 'guest')}")
            self.__client.update_subscriptions({uuid: {'media': {
                "camera": "subscribed",
                "microphone": "unsubscribed"}
            }})
            self.__client.set_video_renderer(
                participant['id'], self.on_video_frame, "camera", "RGB")

        print(f"DONE WITH participants: {participants}")

        # wait for the fetch thread to finish, which it never will as long as
        # no errors are encountered sending audio, so this just keeps us in
        # the session
        self.__send_thread.join()

    def leave(self):
        self.__app_quit = True
        self.__client.leave()

    def maybe_start(self):
        if self.__app_error:
            self.__start_event.set()

        if self.__app_inputs_updated and self.__app_joined:
            self.__start_event.set()

    def fetch_audio(self, text):
        print(f"fetching audio via deepgram -> {self.__text}")
        dg_key = os.environ.get('DEEPGRAM_KEY')
        base_url = 'https://api.beta.deepgram.com/v1/speak'
        voice = 'alpha-asteria-en-v2'
        request_url = f'{base_url}?model={voice}&encoding=linear16&container=none'
        headers = {'authorization': f'token {dg_key}'}

        r = requests.post(request_url, headers=headers,
                          data=text.encode('utf-8'))
        print(
            f'audio fetch status code: {r.status_code}, content length: {len(r.content)}')
        self.__raw_audio_buffer.add_data(r.content)

    def send_raw_audio(self):
        self.__start_event.wait()

        if self.__app_error:
            print(f"Unable to send audio!")
            return

        while not self.__app_quit:
            # fetch 100ms worth of audio
            audio_chunk = self.__raw_audio_buffer.get_data()
            if audio_chunk is None:
                time.sleep(0.01)
                continue
            self.__virtual_mic.write_frames(audio_chunk)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--meeting", required=True,
                        help="Daily meeting URL")
    parser.add_argument("-t", "--text", required=True,
                        help="text to turn into audio")
    args = parser.parse_args()

    Daily.init()

    app = AudioApp()

    try:
        app.run(args.meeting, args.text)
    except KeyboardInterrupt:
        print("Ctrl-C detected. Exiting!")
    finally:
        app.leave()

    time.sleep(2)


if __name__ == '__main__':
    main()
