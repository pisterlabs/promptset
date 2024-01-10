import io
from typing import Any, Generator
import soundcard as sc
import numpy as np
from numpy.typing import NDArray
from numpy import float32
from multiprocessing import Process, Queue
from openai import OpenAI
import soundfile as sf

INTERVAL = 3
BUFFER_SIZE = 4096
b = np.ones(100) / 100
SAMPLE_RATE = 16000
CHANNELS = 1

# TODO 設定ファイルで調整できるようにしたい。
THRESHOLD = 0.09


def record(microphone) -> Generator[NDArray[np.float32], Any, Any]:
    global SAMPLE_RATE, CHANNELS
    with microphone.recorder(samplerate=SAMPLE_RATE, channels=CHANNELS) as recorder:
        while True:
            data = recorder.record(BUFFER_SIZE)
            yield data


def record_speaker():
    mic = sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)
    yield from record(mic)


def record_microphone():
    mic = sc.default_microphone()
    yield from record(mic)


def record_speaker_multi(q):
    for peace in record_speaker():
        q.put(peace)


def mix_audio(input1: NDArray[float32], input2: NDArray[float32]):
    return input1 + input2


def record_microphone_multi(q):
    for peace in record_microphone():
        q.put(peace)


def record_mix_audio():
    speakerQueue = Queue()
    micQueue = Queue()
    speakerProcess = Process(target=record_speaker_multi, args=(speakerQueue,))
    micProcess = Process(target=record_microphone_multi, args=(micQueue,))
    speakerProcess.start()
    micProcess.start()
    try:
        while True:
            newSpeakerAudio = speakerQueue.get()
            newMicAudio = micQueue.get()
            yield mix_audio(newSpeakerAudio, newMicAudio)
    except KeyboardInterrupt:
        speakerProcess.terminate()
        micProcess.terminate()
        raise
    finally:
        speakerProcess.terminate()
        micProcess.terminate()


def sample_length(second: int):
    return second * SAMPLE_RATE


def slice_by_silence(
    stream: Generator[NDArray[float32], Any, Any]
) -> Generator[NDArray[float32], Any, Any]:
    global SAMPLE_RATE, CHANNELS
    n = 0
    audio = np.empty(SAMPLE_RATE * INTERVAL + BUFFER_SIZE, dtype=np.float32)

    try:
        while True:
            while n < SAMPLE_RATE * INTERVAL:
                data = next(stream)
                last = n + len(data)
                audio[n:last] = data.reshape(-1)
                n += len(data)
            # find silent periods
            m = n * 4 // 5
            vol = np.convolve(audio[m:n] ** 2, b, "same")
            m += vol.argmin()
            yield audio[:m]
            # 前周の差分残し
            audio_prev = audio
            audio = np.empty(SAMPLE_RATE * INTERVAL + BUFFER_SIZE, dtype=np.float32)
            audio[: n - m] = audio_prev[m:n]
            n = n - m
    except KeyboardInterrupt:
        yield audio[:n]
        raise


def slice_by_seconds(
    seconds: int, stream: Generator[NDArray[float32], Any, Any]
) -> Generator[NDArray[float32], Any, Any]:
    try:
        sample_length = seconds * SAMPLE_RATE
        data = np.empty(0, dtype=np.float32)
        for peace in stream:
            data = np.append(data, peace)
            if len(data) >= sample_length:
                yield data
                data = np.empty(0, dtype=np.float32)
    except KeyboardInterrupt:
        yield data
        raise


def isIncludeVoice(audio: NDArray[float32]):
    # 無音をwhisperに送るとバグるので発話検知
    # ひとまず音量の最大値で判定
    # TODO より精密な発話検知。例えば連続した音の長さが0.5秒以上あるゾーンが一つでもあれば、とか。
    # ノイズフィルタあてる？それだと音質微妙なマイク使ってる人いたらやばいかも
    # ポップノイズだけカットした上で、音量の最大値で判定するのがいいかも

    # min = np.min(np.abs(audio))
    maxVal = np.max(np.abs(audio))
    # print(min, max)
    return maxVal > THRESHOLD


def speechToText(audio: NDArray[float32], language: str, prompt: str | None = None):
    # ここで処理してるときにKeyboardInterruptした時の挙動調整
    if isIncludeVoice(audio):
        buffer = io.BytesIO()
        sf.write(buffer, audio, SAMPLE_RATE, format="wav")
        buffer.name = "output.wav"
        openai = OpenAI()
        result = openai.audio.transcriptions.create(
            model="whisper-1", file=buffer, language=language, prompt=prompt
        )
        return result.text
    else:
        return ""


def record_audio_limited() -> NDArray[float32]:
    global SAMPLE_RATE, CHANNELS, INTERVAL
    mic = sc.default_microphone()
    with mic.recorder(samplerate=SAMPLE_RATE, channels=CHANNELS) as recorder:
        n = 0
        audio = np.empty(SAMPLE_RATE * INTERVAL + BUFFER_SIZE, dtype=np.float32)
        while n < SAMPLE_RATE * INTERVAL:
            data = recorder.record(BUFFER_SIZE)
            last = n + len(data)
            audio[n:last] = data.reshape(-1)
            n += len(data)
        return audio


def record_audio2(stream):
    global SAMPLE_RATE, CHANNELS
    n = 0
    audio = np.empty(SAMPLE_RATE * INTERVAL + BUFFER_SIZE, dtype=np.float32)

    try:
        while True:
            while n < SAMPLE_RATE * INTERVAL:
                data = next(stream)
                last = n + len(data)
                audio[n:last] = data.reshape(-1)
                n += len(data)
            # find silent periods
            m = n * 4 // 5
            vol = np.convolve(audio[m:n] ** 2, b, "same")
            m += vol.argmin()
            yield audio[:m]
            # 前周の差分残し
            audio_prev = audio
            audio = np.empty(SAMPLE_RATE * INTERVAL + BUFFER_SIZE, dtype=np.float32)
            audio[: n - m] = audio_prev[m:n]
            n = n - m
    except KeyboardInterrupt:
        yield audio[:n]
        raise


def record_audio(sample_rate: int, channels: int):
    speaker = sc.get_microphone(
        id=str(sc.default_speaker().name), include_loopback=True
    )
    with speaker.recorder(samplerate=sample_rate, channels=channels) as recorder:
        audio = np.empty(sample_rate * INTERVAL + BUFFER_SIZE, dtype=np.float32)
        n = 0
        try:
            while True:
                while n < sample_rate * INTERVAL:
                    data = recorder.record(BUFFER_SIZE)
                    last = n + len(data)
                    audio[n:last] = data.reshape(-1)
                    n += len(data)

                # find silent periods
                m = n * 4 // 5
                vol = np.convolve(audio[m:n] ** 2, b, "same")
                m += vol.argmin()
                yield audio[:m]
                # 前周の差分残し
                audio_prev = audio
                audio = np.empty(sample_rate * INTERVAL + BUFFER_SIZE, dtype=np.float32)
                audio[: n - m] = audio_prev[m:n]
                n = n - m
        except KeyboardInterrupt:
            yield audio[:n]
            raise
