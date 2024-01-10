# Copyright 2023 Ole Kliemann
# SPDX-License-Identifier: GPL-3.0-or-later

import asyncio
from pydub import AudioSegment
import tempfile
from openai import OpenAI
from mreventloop import emits, slot, has_event_loop, forwards
from drdictaphone.pipeline_events import PipelineEvents, PipelineSlots
from drdictaphone.config import config
from drdictaphone import logger
logger = logger.get(__name__)

@has_event_loop('event_loop')
@forwards(PipelineSlots)
@emits('events', PipelineEvents)
class Transcriber:
  cost_second = (0.6 / 60)

  def __init__(self, language):
    self.language = language
    self.context = []

  def transcribeBuffer(self, audio):
    with tempfile.NamedTemporaryFile(
      prefix = "recorded_audio_",
      suffix = ".mp3",
      delete = True
    ) as temp_file:
      audio.export(temp_file.name, format = 'mp3')
      logger.debug(f'audio file: {temp_file.name}')
      audio_file = open(temp_file.name, 'rb')
      client = OpenAI(api_key = config['openai_api_key'])
      transcript = client.audio.transcriptions.create(
        model = "whisper-1",
        file = audio_file,
        language = self.language,
        prompt = ' '.join(self.context)
      )
      return transcript.text

  @slot
  async def onResult(self, audio):
    logger.debug(f'received audio of length: {len(audio)}')
    text = await asyncio.get_event_loop().run_in_executor(None, self.transcribeBuffer, audio)
    logger.debug(f'whisper replied: {text}')
    logger.debug(f'context was: {self.context}')
    length_seconds = len(audio) / 1000
    costs = length_seconds * Transcriber.cost_second
    logger.debug(f'costs: {costs}')
    self.events.result(text)
    self.events.costs_incurred(costs)
    self.context.append(text)

  @slot
  def onClearBuffer(self):
    self.context = []
    self.events.clear_buffer()
