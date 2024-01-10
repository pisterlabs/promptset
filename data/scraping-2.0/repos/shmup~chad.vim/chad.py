#!/usr/bin/env python3
# ==============================================================================
# buffer interaction with openai chat completion
# file:     chad.py
# author:   shmup <https://github.com/shmup>
# website:  https://github.com/shmup/chad.vim
# updated:  dec-24-2023
# license:  :h license
# ==============================================================================

import json
import os
import signal
from openai import OpenAI

# pyright: reportUndefinedVariable=false, reportGeneralTypeIssues=false


class InterruptedError(Exception):
  pass


def signal_handler(_, __):
  raise InterruptedError()


signal.signal(signal.SIGINT, signal_handler)


class ChatInterface:

  def __init__(self, vim, api_key, model):
    self.vim = vim
    self.client = OpenAI(api_key=api_key)
    self.model = model

  def fetch_response(self, messages, options):
    if not self.client:
      raise ValueError("client is not set up.")

    assert hasattr(self.client, 'chat'), "client does not have a 'chat' attr"

    params = {
        "model": self.model,
        "messages": messages,
        "stream": options.get('stream', True),
        "temperature": float(options.get('temperature', 1)),
        "presence_penalty": float(options.get('presence_penalty', 0)),
        "frequency_penalty": float(options.get('frequency_penalty', 0)),
    }
    return self.client.chat.completions.create(**params)

  def parse_buffer(self, buffer_content):
    messages, role, content = [], None, []
    for line in buffer_content.splitlines():
      if line.startswith("### "):
        if role:
          messages.append({"role": role, "content": "\n".join(content).strip()})
        role, content = line[4:].lower(), []
      else:
        content.append(line)
    if role:
      messages.append({"role": role, "content": "\n".join(content).strip()})
    return messages

  def run(self, openai_options):
    buffer_content = self.vim.eval('join(getline(1, "$"), "\n")')
    messages = self.parse_buffer(buffer_content)
    try:
      response = self.fetch_response(messages, openai_options)
      self.vim.command('call append("$", "### assistant")')
      self.update_buffer(response)
    except (InterruptedError, KeyboardInterrupt):
      self.vim.command('echo "Chad cancelled"')
    except Exception as e:
      error_message = str(e)
      self.vim.command(f'call append("$", "Error: {error_message}")')
    finally:
      self.vim.command('call append("$", "### user")\nnormal G')

  def update_buffer(self, stream):
    buffer = self.vim.current.buffer
    line_num = len(buffer) - 1

    for chunk in stream:
      if self._should_skip_chunk(chunk):
        continue
      text = self._normalize_text(chunk.choices[0].delta.content)
      lines = text.split('\n')
      line_num = self._update_buffer_lines(buffer, line_num, lines)
      self._refresh_display()

  def _should_skip_chunk(self, chunk):
    return chunk.choices[0].delta is None or chunk.choices[
        0].delta.content is None

  def _normalize_text(self, content):
    return content.replace('\r', '')

  def _update_buffer_lines(self, buffer, line_num, lines):
    if buffer[line_num].startswith("### "):
      line_num += 1
      buffer[line_num:line_num] = lines
    else:
      buffer[line_num] += lines[0]
      if len(lines) > 1:
        buffer[line_num + 1:line_num + 1] = lines[1:]
    return line_num + len(lines) - 1

  def _refresh_display(self):
    self.vim.command("redraw")
    self.vim.command("normal G")


def main():
  api_key = os.getenv('CHAD')
  if not api_key:
    raise ValueError("please export CHAD=api_key_here")

  openai_options = json.loads(vim.eval('g:openai_options'))
  chat_interface = ChatInterface(vim, openai_options['api_key'],
                                 openai_options['model'])
  chat_interface.run(openai_options)


if __name__ == '__main__':
  main()
