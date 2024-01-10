#!/usr/bin/env -S python3

import os
import random
import openai

def generate_jargon(title):
  openai.api_key = os.getenv("OPENAI_API_KEY")

  response = openai.Completion.create(
    model="text-davinci-003",
    prompt="masterhackerbot is a bot on Reddit that responds with random tech <jargon> which will be automatically replaced later.\nThe <jargon> is a very long verb phrase and should be followed by a comma and a dependent clause.\n\nExample: To perform this action, you must <jargon>, which will allow you to perform it.\n\nTitle: \"Epic TikTok biohacker\"\nmasterhackerbot: To epicly hack TikTok, you have to <jargon>, allowing you to be a TikTok biohacker.\n\nTitle: \"" + title + "\"\nmasterhackerbot:",
    temperature=0.32,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=["\n"]
  )
  #print(response)
  choice = response.get("choices", 0)
  generated = choice[0].get("text", "An error has occurred. Error code 01189998819991197253")
  generated = generated[1:]

  jargon_list = [
  ["TCP", "HTTP", "SDD", "RAM", "GB", "CSS", "SSL", "AGP", "SQL", "FTP", "PCI", "AI", "ADP", "RSS", "XML", "EXE", "COM", "HDD", "THX", "SMTP", "SMS", "USB", "PNG", "PHP", "UDP", "TPS", "RX", "ASCII", "CD-ROM", "CGI", "CPU", "DDR", "DHCP", "BIOS", "IDE", "IP", "MAC", "MP3", "AAC", "PPPoE", "SSD", "SDRAM", "VGA", "XHTML", "Y2K", "GUI"],
  ["auxiliary", "primary", "back-end", "digital", "open-source", "virtual", "cross-platform", "redundant", "online", "haptic", "multi-byte", "bluetooth", "wireless", "1080p", "neural", "optical", "solid state", "mobile", "unicode", "backup", "high speed", "56k", "analog", "fiber optic", "central", "visual", "ethernet"],
  ["driver", "protocol", "bandwidth", "panel", "microchip", "program", "port", "card", "array", "interface", "system", "sensor", "firewall", "hard drive", "pixel", "alarm", "feed", "monitor", "application", "transmitter", "bus", "circuit", "capacitor", "matrix", "address", "form factor", "array", "mainframe", "processor", "antenna", "transistor", "virus", "malware", "spyware", "network", "internet"],
  ["back up", "bypass", "hack", "override", "compress", "copy", "navigate", "index", "connect", "generate", "quantify", "calculate", "synthesize", "input", "transmit", "program", "reboot", "parse", "shut down", "inject", "transcode", "encode", "attach", "disconnect", "network"],
  ["backing up", "bypassing", "hacking", "overriding", "compressing", "copying", "navigating", "indexing", "connecting", "generating", "quantifying", "calculating", "synthesizing", "inputting", "transmitting", "programming", "rebooting", "parsing", "shutting down", "injecting", "transcoding", "encoding", "attaching", "disconnecting", "networking"]
  ]

  here = os.path.dirname(os.path.abspath(__file__))
  filename = os.path.join(here, 'phrases.txt')

  txt_file = open(filename, "r")
  file_content = txt_file.read()
  phrases = file_content.split("\n\n")

  comment_temp = (phrases[random.randint(0,len(phrases)-1)])

  #print(comment_temp)

  sentence = comment_temp

  # split the sentence into a list of words and phrases separated by commas
  phrases = sentence.split(',')

  # go through each phrase in the list
  for i in range(len(phrases)):
    # split the phrase into a list of words
    words = phrases[i].split()
    # go through each word in the list
    for j in range(len(words)):
      # check if the word is a placeholder
      if words[j][0] == '[' and words[j][-1] == ']':
        # get the index of the placeholder
        index = int(words[j][1:-1])
        # get a random word from the corresponding list
        replacement = random.choice(jargon_list[index])
        # replace the placeholder with the word
        words[j] = replacement
    # join the list of words back into a phrase
    phrases[i] = " ".join(words)

  # join the list of phrases back into a sentence
  jargon_phrase = ", ".join(phrases)

  complete = generated.replace("<jargon>", jargon_phrase)

  return complete

def skidquestion():
  openai.api_key = os.getenv("OPENAI_API_KEY")

  response = openai.Completion.create(
    model="text-davinci-003",
    prompt="input: I want you to ask questions a script kiddie would ask, for example, how install kali linucks. Pick one and format your response as <script kiddie question>. Do not add any additional formatting or text. Format as plain text with no special symbols. output: ",
    temperature=0.32,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=["input: "]
  )
  return response.choices[0].text

if __name__ == '__main__':
  title = input("Enter the title of the post... ")
  result = generate_jargon(title)
  print(result)