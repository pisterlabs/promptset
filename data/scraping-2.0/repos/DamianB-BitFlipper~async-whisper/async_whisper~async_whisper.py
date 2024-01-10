import asyncio
import io
import time
from dataclasses import dataclass
from typing import cast

import openai
from aiolimiter import AsyncLimiter
from pydub import AudioSegment

from .logger import logger
from .stitch_utils import resolve_overlap, stitch_audio_segments


@dataclass
class _AudioChunk:
    segment: AudioSegment
    segment_length_ms: int
    transcription: str | None = None

    @property
    def transcription_words(self) -> list[str]:
        if self.transcription is None:
            raise ValueError("Transcription is not set")
        return self.transcription.split()


class Defaults:
    # Allow a maximum of 100 requests per minute
    ASYNC_RATE_LIMIT_RPM = 100

    # Timeout and retry after 15 seconds for segment transcription
    TRANSCRIBE_SEGMENT_TIMEOUT = 15

    # Each segment is 60 seconds long
    SEGMENT_LENGTH_MS = 60_000

    # Have a 10 second overlap between each segment
    OVERLAP_LENGTH_MS = 10_000

    # The default language is English
    LANGUAGE = "en"

    # When stitching together transcription segments, have
    # a `STITCH_WIGGLE` of words wiggle room
    STITCH_WIGGLE = 15

    # How many words in a row must be identical before we start
    # picking from the following segment during overlap resolution
    RESOLVE_OVERLAP_THRESHOLD = 4


class AsyncWhisper:
    def __init__(
        self,
        openai_api_key: str,
        *,
        audio_chunk_ms: int = Defaults.SEGMENT_LENGTH_MS,
        overlap_ms: int = Defaults.OVERLAP_LENGTH_MS,
        rate_limit_rpm: int = Defaults.ASYNC_RATE_LIMIT_RPM,
        retry_timeout: int | None = Defaults.TRANSCRIBE_SEGMENT_TIMEOUT,
        language: str = Defaults.LANGUAGE,
        sitch_wiggle: int = Defaults.STITCH_WIGGLE,
        resolve_overlap_threshold: int = Defaults.RESOLVE_OVERLAP_THRESHOLD,
    ):
        # Save the values to the instance
        self.openai_api_key = openai_api_key
        self.audio_chunk_ms = audio_chunk_ms
        self.overlap_ms = overlap_ms
        self.rate_limit_rpm = rate_limit_rpm
        self.retry_timeout = retry_timeout
        self.language = language
        self.stitch_wiggle = sitch_wiggle
        self.resolve_overlap_threshold = resolve_overlap_threshold

        # Create an async OpenAI `client`
        self.client = openai.AsyncOpenAI(
            api_key=self.openai_api_key,
        )

        # Create an `AsyncLimiter` to limit the rate of requests
        self.rate_limiter = AsyncLimiter(self.rate_limit_rpm, 60)

    async def _transcribe_audio_segment(
        self,
        audio_segment: AudioSegment,
        *,
        uid: int,
        prompt: str,
    ) -> str:
        logger.info(f"{uid:3}: Starting transcription...")

        # Load the `audio_segment` into a buffer
        buffer = io.BytesIO()
        audio_segment.export(buffer, format="mp3")

        # Trick OpenAI into thinking the `buffer` is an mp3 file
        buffer.name = "audio_segment.mp3"

        start_time = time.time()
        retry_timeout = self.retry_timeout

        # Retry the request until it succeeds
        while True:
            try:
                transcript = await asyncio.wait_for(
                    self.client.audio.transcriptions.create(
                        file=buffer,
                        model="whisper-1",
                        language=self.language,
                        prompt=prompt,
                    ),
                    timeout=retry_timeout,
                )
                break
            except asyncio.TimeoutError:
                # Sanity check
                assert retry_timeout is not None

                # Backoff the `retry_timeout` for the next request
                retry_timeout *= 2

                logger.warning("Timeout error, retrying...")
            except (
                openai.APIConnectionError,
                openai.APIStatusError,
                openai.RateLimitError,
            ) as e:
                logger.warning(
                    f"An error occurred processing audio segment: {e}, retrying in 5 seconds...",
                )
                await asyncio.sleep(5)

        logger.info(f"{uid:3}: Transcribed in {time.time() - start_time} seconds")

        return transcript.text

    async def _safe_transcribe_audio_segment(
        self,
        audio_segment: AudioSegment,
        *,
        uid: int,
        prompt: str = "",
    ) -> str:
        async with self.rate_limiter:
            return await self._transcribe_audio_segment(
                audio_segment,
                uid=uid,
                prompt=prompt,
            )

    async def _transcribe_audio_chunks(
        self, audio_chunks: list[_AudioChunk]
    ) -> list[str]:
        start_time = time.time()
        # Transcribe each segment in `segments`
        transcription_tasks = [
            self._safe_transcribe_audio_segment(
                audio_chunk.segment,
                uid=audio_chunk_id,
            )
            for audio_chunk_id, audio_chunk in enumerate(audio_chunks)
        ]
        transcriptions = await asyncio.gather(*transcription_tasks)

        logger.info(f"Transcribed all chunks in {time.time() - start_time} seconds")

        return transcriptions

    def _chunk_audio(self, audio_segment: AudioSegment) -> list[_AudioChunk]:
        audio_chunks = []
        total_length = len(audio_segment)
        start = 0
        while True:
            # Make `self.audio_chunk_ms` segments
            end = min(start + self.audio_chunk_ms, total_length)

            # Add the segment to the list
            audio_chunks.append(
                _AudioChunk(
                    # Indexing an AudioSegment returns a strange type
                    segment=cast(AudioSegment, audio_segment[start:end]),
                    segment_length_ms=end - start,
                )
            )

            # Break if we're at the end of the audio segment
            if end == total_length:
                break

            # Increment the start time
            start += self.audio_chunk_ms - self.overlap_ms

        return audio_chunks

    def _stitch_together_words(
        self,
        before_words: list[str],
        before_length_ms: int,
        after_words: list[str],
        after_length_ms: int,
    ) -> list[str]:
        # Approximate the overlap length by extrapolating the words spoken per second
        # from the `before_words` and the `after_words`
        approx_overlap_len = int(
            (len(before_words) + len(after_words))
            * (self.overlap_ms / (before_length_ms + after_length_ms))
        )

        stitch_meta = stitch_audio_segments(
            before_words=before_words,
            after_words=after_words,
            approx_overlap_len=approx_overlap_len,
            stitch_wiggle=self.stitch_wiggle,
        )

        stitch_str1_words = before_words[: -stitch_meta.overlap_len]
        stitch_str2_words = after_words[stitch_meta.overlap_len :]
        stitch_overlap_words = resolve_overlap(
            overlap1=before_words[-stitch_meta.overlap_len :],
            overlap2=after_words[: stitch_meta.overlap_len],
            streak_threshold=self.resolve_overlap_threshold,
        )

        # Combine the two stitches
        stitch_words = stitch_str1_words + stitch_overlap_words + stitch_str2_words

        return stitch_words

    async def transcribe_audio(self, audio: AudioSegment) -> str:
        audio_chunks = self._chunk_audio(audio)

        # Transcribe each of the `audio_chunks`
        transcriptions = await self._transcribe_audio_chunks(audio_chunks)

        # Set the `transcription` attribute of each `AudioChunk`
        for audio_chunk, transcription in zip(audio_chunks, transcriptions):
            audio_chunk.transcription = transcription

        # Stitch the transcription segments together
        acc_words = audio_chunks[0].transcription_words
        for i in range(1, len(audio_chunks)):
            prev_audio_chunk = audio_chunks[i - 1]

            current_audio_chunk = audio_chunks[i]
            current_words = current_audio_chunk.transcription_words

            stitch_words = self._stitch_together_words(
                before_words=acc_words,
                before_length_ms=prev_audio_chunk.segment_length_ms,
                after_words=current_words,
                after_length_ms=current_audio_chunk.segment_length_ms,
            )

            # Update the `acc_words` for the next iteration
            acc_words = stitch_words

        # The stitched transcript is the final `acc_words`
        stitched_transcript = " ".join(acc_words)

        return stitched_transcript
