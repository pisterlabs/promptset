# Kitt 1.5
from elevenlabs import generate, stream, TTS, Voice, VoiceSettings, Model
from websockets.sync.client import connect as client_ws_connect
from logging import basicConfig, INFO  # info, warn, error,
from typing import Callable, Dict, Iterator, Union
from quart.typing import ResponseReturnValue
from logging.config import dictConfig
from signal import SIGINT, SIGTERM
from quart import Quart, websocket
from dotenv import load_dotenv
from os import environ as env
from deepgram import Deepgram
from livekit import api, rtc
from openai import OpenAI
from quart import request
from pathlib import Path
import numpy as np
import websockets
import asyncio
import json

load_dotenv()
app = Quart(__name__)

# ******** ENV VARIABLES  ********

LIVEKIT_API_KEY = env.get("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = env.get("LIVEKIT_API_SECRET")
LIVEKIT_WS_URL = env.get("LIVEKIT_WS_URL")

OPENAI_API_KEY = env.get("OPENAI_API_KEY")
OPENAI_ORG_ID = env.get("OPENAI_ORG_ID")

ELEVENLABS_API_KEY = env.get("ELEVENLABS_API_KEY")
DEEPGRAM_API_KEY = env.get("DEEPGRAM_API_KEY")

# ******** ENSURE ENV VARIABLES ARE SET ********
assert isinstance(LIVEKIT_API_KEY, str), "LIVEKIT_API_KEY must be set!"
assert isinstance(LIVEKIT_API_SECRET, str), "LIVEKIT_API_SECRET must be set!"
assert isinstance(LIVEKIT_WS_URL, str), "LIVEKIT_WS_URL must be set!"
assert isinstance(OPENAI_API_KEY, str), "OPENAI_API_KEY must be set!"
assert isinstance(OPENAI_ORG_ID, str), "OPENAI_ORG_ID must be set!"
assert isinstance(ELEVENLABS_API_KEY, str), "ELEVENLABS_API_KEY must be set!"
assert isinstance(DEEPGRAM_API_KEY, str), "DEEPGRAM_API_KEY must be set!"

# ******** GLOBAL VARIABLES  ********
SAMPLE_RATE = 48000
SAMPLES_PER_CHANNEL = 480  # 10ms at 48kHz
NUM_CHANNELS = 1
BOT_NAME = "talking_donut"
VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
SPLITTERS = [".", ",", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]", "}", " "]
WHISPER_SAMPLE_RATE = 16000
# *********** CLIENTS ***********

OPEN_AI_CLIENT = OpenAI(
    api_key=OPENAI_API_KEY,
    organization=OPENAI_ORG_ID,
)

DEEPGRAM_CLIENT = Deepgram(DEEPGRAM_API_KEY)

DEEPGRAM_PARAMS = {
            "punctuate": True, 
            "numerals": True,
            "model": "general", 
            "language": "en-US",
            "tier": "enhanced" 
          }

DEEPGRAM_TIME_LIMT = float('inf')
DEEPGRAM_TRANSCRIPT_ONLY = True


def text_chunker(chunks: Iterator[str]) -> Iterator[str]:
    """Used during input streaming to chunk text blocks and set last char to space"""
    splitters = (".", ",", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]", "}", " ")
    buffer = ""
    for text in chunks:
        if buffer.endswith(splitters):
            yield buffer if buffer.endswith(" ") else buffer + " "
            buffer = text
        elif text.startswith(splitters):
            output = buffer + text[0]
            yield output if output.endswith(" ") else output + " "
            buffer = text[1:]
        else:
            buffer += text
    if buffer != "":
        yield buffer + " "
        
def generate_stream_input(
        text: Iterator[str], voice: Voice
    ) -> Iterator[bytes]:
        BOS = json.dumps(
            dict(
                text=" ",
                try_trigger_generation=True,
                voice_settings=voice.settings.model_dump() if voice.settings else None,
                generation_config=dict(
                    chunk_length_schedule=[50],
                ),
            )
        )
        EOS = json.dumps(dict(text=""))

        with client_ws_connect(
            f"wss://api.elevenlabs.io/v1/text-to-speech/{voice.voice_id}/stream-input?model_id=eleven_monolingual_v2",
            additional_headers={"xi-api-key": ELEVENLABS_API_KEY}
        ) as websocket:
            # Send beginning of stream
            websocket.send(BOS)
            app.logger.info("CONNECTED AND STARTED ELEVEN LABS WS CONNECTION")

            # Stream text chunks and receive audio
            for text_chunk in text_chunker(text):
                data = dict(text=text_chunk, try_trigger_generation=True)
                websocket.send(json.dumps(data))
                print("\n\nSENT TO ELEVEN LABS")
                try:
                    data = json.loads(websocket.recv(1e-4))
                    if data["audio"]:
                        yield base64.b64decode(data["audio"])  # type: ignore
                except TimeoutError:
                    pass

            # Send end of stream
            websocket.send(EOS)

            # Receive remaining audio
            while True:
                try:
                    data = json.loads(websocket.recv())
                    if data["audio"]:
                        yield base64.b64decode(data["audio"])  # type: ignore
                except websockets.exceptions.ConnectionClosed:
                    break

def ends_with_splitter(chunk):
    splitters = [".", ",", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]", "}", " "]
    return (
        chunk
        and chunk != " "
        and any(chunk.endswith(splitter) for splitter in splitters)
    )

async def audio_chunks_to_livekit(audio_stream: bytes | Iterator[bytes], audio_source: rtc.AudioSource,):
    audio_frame = rtc.AudioFrame.create(SAMPLE_RATE, NUM_CHANNELS, SAMPLES_PER_CHANNEL)
    audio_data = np.frombuffer(audio_frame.data, dtype=np.int16)

    if isinstance(audio_stream, bytes):
        np.copyto(audio_data, audio_stream)
        await audio_source.capture_frame(audio_frame)
    else:
        for chunk in audio_stream:
            if chunk is not None:
                np.copyto(audio_data, chunk)
                await audio_source.capture_frame(audio_frame)            

async def chat_completion(
    text_data: str,
    room: rtc.Room,
    participant,
    audio_source: rtc.AudioSource,
):
    try:
        gpt_resp = OPEN_AI_CLIENT.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text_data},
            ],
            stream=True,
        )

        def gpt_text_resp_iter() -> Iterator[str]:
            chat_resp_chunk: str = ""
            for chunk in gpt_resp:
                msg_chunk = chunk.choices[0].delta.content
                is_finished = chunk.choices[0].finish_reason
                if is_finished:
                    break
                else:
                    if msg_chunk:
                        chat_resp_chunk += msg_chunk
                    if ends_with_splitter(chat_resp_chunk):
                        app.logger.info(f"chat_resp_chunk - [{chat_resp_chunk}]")
                        yield chat_resp_chunk
                        chat_resp_chunk: str = ""
        
        gpt_text_resp_iter()
        
        # voice = Voice(
        #     voice_id=VOICE_ID,
        #     name="Bella",
        #     settings= VoiceSettings(
        #         stability=0.71, similarity_boost=0.5, style=0.0, use_speaker_boost=True
        #     ),
        # )
        
        # audio_stream = generate_stream_input(gpt_text_resp_iter(), voice)
        # await audio_chunks_to_livekit(audio_stream, audio_source)

    except Exception as e:
        app.logger.error(f"Exception while processing data received: {e}")

async def handle_audio(audio_stream: rtc.AudioStream, deepgramLive):
    async for frame in audio_stream:
        frame = frame.remix_and_resample(WHISPER_SAMPLE_RATE, 1)
        audio_bytes = frame.data.tobytes()
        deepgramLive.send(audio_bytes)
        

async def ai_bot(room: rtc.Room, participant: str) -> None:
    # publish a track
    audio_source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
    track = rtc.LocalAudioTrack.create_audio_track(f"{BOT_NAME}_voice", audio_source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE
    audio_publication = await room.local_participant.publish_track(track, options)
    app.logger.info("published audio track %s", audio_publication.sid)

    try:
        deepgramLive = await DEEPGRAM_CLIENT.transcription.live(DEEPGRAM_PARAMS) # type: ignore
    except Exception as e:
        print(f'Could not open socket: {e}')
        return
    # Listen for the connection to close
    deepgramLive.registerHandler(deepgramLive.event.CLOSE,  # type: ignore
                                 lambda _: print('✅ Transcription complete! Connection closed. ✅'))

    
    
    @room.on("data_received")
    def on_data_received(
        data: bytes, kind: rtc.DataPacketKind, participant: rtc.Participant
    ):
        msg_dict = json.loads(data)
        raw_msg: str = msg_dict.get("message")
        app.logger.info("received data from %s: %s", participant.identity, msg_dict)
        if raw_msg:
            asyncio.create_task(
                chat_completion(raw_msg, room, participant, audio_source)
            )
        else:
            app.logger.warning("didn't send data")

    #  ************* OTHERS *******************

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
        app.logger.info(
            "participant connected: %s %s", participant.sid, participant.identity
        )

    @room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        app.logger.info(
            "participant disconnected: %s %s", participant.sid, participant.identity
        )

    @room.on("local_track_published")
    def on_local_track_published(
        publication: rtc.LocalTrackPublication,
        track: Union[rtc.LocalAudioTrack, rtc.LocalVideoTrack],
    ):
        app.logger.info("local track published: %s", publication.sid)

    @room.on("active_speakers_changed")
    def on_active_speakers_changed(speakers: list[rtc.Participant]):
        app.logger.info("active speakers changed: %s", speakers)

    @room.on("local_track_unpublished")
    def on_local_track_unpublished(publication: rtc.LocalTrackPublication):
        app.logger.info("local track unpublished: %s", publication.sid)

    @room.on("track_published")
    def on_track_published(
        publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ):
        app.logger.info(
            "track published: %s from participant %s (%s)",
            publication.sid,
            participant.sid,
            participant.identity,
        )

    @room.on("track_unpublished")
    def on_track_unpublished(
        publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ):
        app.logger.info("track unpublished: %s", publication.sid)

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        app.logger.info("track subscribed: %s", publication.sid)
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            _video_stream = rtc.VideoStream(track)
            # video_stream is an async iterator that yields VideoFrame
        elif track.kind == rtc.TrackKind.KIND_AUDIO:
            print("Subscribed to an Audio Track")
            # audio_stream is an async iterator that yields AudioFrame
            audio_stream = rtc.AudioStream(track)
            asyncio.create_task(handle_audio(audio_stream, deepgramLive))

    @room.on("track_unsubscribed")
    def on_track_unsubscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        app.logger.info("track unsubscribed: %s", publication.sid)

    @room.on("track_muted")
    def on_track_muted(
        publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ):
        app.logger.info("track muted: %s", publication.sid)

    @room.on("track_unmuted")
    def on_track_unmuted(
        publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ):
        app.logger.info("track unmuted: %s", publication.sid)

    @room.on("connection_quality_changed")
    def on_connection_quality_changed(
        participant: rtc.Participant, quality: rtc.ConnectionQuality
    ):
        app.logger.info("connection quality changed for %s", participant.identity)

    @room.on("track_subscription_failed")
    def on_track_subscription_failed(
        participant: rtc.RemoteParticipant, track_sid: str, error: str
    ):
        app.logger.info("track subscription failed: %s %s", participant.identity, error)

    @room.on("connection_state_changed")
    def on_connection_state_changed(state: rtc.ConnectionState):
        app.logger.info("connection state changed: %s", state)

    @room.on("connected")
    def on_connected() -> None:
        app.logger.info("connected")

    @room.on("disconnected")
    def on_disconnected() -> None:
        app.logger.info("disconnected")

    @room.on("reconnecting")
    def on_reconnecting() -> None:
        app.logger.info("reconnecting")

    @room.on("reconnected")
    def on_reconnected() -> None:
        app.logger.info("reconnected")

    app.logger.info("connected to room %s", room.name)
    app.logger.info("participants: %s", room.participants)

@app.route("/lsdk-webhook", methods=["POST"])
async def handle_livekit_webhook():
    token_verifier = api.TokenVerifier(
        api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET
    )
    webhook_receiver = api.WebhookReceiver(token_verifier)

    jwt = request.headers.get("Authorization")
    if not jwt:
        app.logger.error("no JWT in authorization header")
        return "no JWT in authorization header", 401

    body = await request.get_data(as_text=True)

    try:
        event = webhook_receiver.receive(body, jwt)
        if event.event == "room_started":
            participant_room = event.room
            participant = event.participant.name
            room_name = participant_room.name

            app.logger.info("participant in the room {participant}")

            token = (
                api.AccessToken(api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)
                .with_identity(BOT_NAME)
                .with_name(BOT_NAME)
                .with_grants(
                    api.VideoGrants(
                        room=room_name,
                        room_join=True,
                        room_list=True,
                        room_record=True,
                        can_update_own_metadata=True,
                    )
                )
                .to_jwt()
            )

            room = rtc.Room()
            await room.connect(
                LIVEKIT_WS_URL, token, options=rtc.RoomOptions(auto_subscribe=True)
            )
            app.logger.info("connected to room %s", room.name)
            asyncio.create_task(ai_bot(room, participant))

            # if num_participants < max_participants
        else:
            app.logger.info(f"Received event: {event.event}")
        return "Webhook processed 🎉", 200
    except Exception as e:
        app.logger.error(f"Error processing webhook: {e}")
        return str(e), 500


@app.route("/", methods=["GET"])
def health_check() -> ResponseReturnValue:
    return "Server is Up ", 200


# @app.after_serving
# async def create_db_pool():
#     deepgramLive = app.deepgramLive
#     if deepgramLive:
#         await deepgramLive.finish()
        
if __name__ == "__main__":
    dictConfig(
        {
            "version": 1,
            "loggers": {
                "quart.app": {
                    "level": "ERROR",
                },
            },
        }
    )
    # basicConfig(level=INFO)
    # handlers=[logging.FileHandler("server.log"), logging.StreamHandler()],
    port = int(env.get("PORT", 6669))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
