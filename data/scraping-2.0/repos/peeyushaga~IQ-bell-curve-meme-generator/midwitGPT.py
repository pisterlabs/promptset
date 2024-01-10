import wget
import os
from PIL import Image, ImageDraw, ImageFont, ImageOps
from IPython.display import display
import warnings
import cohere
from simplistic_prompt import generate_simplistic_prompt
from midwit_prompt import generate_midwit_prompt

# wget.download("https://ultralytics.com/assets/Arial.ttf", "arial.ttf")

warnings.filterwarnings("ignore", category=DeprecationWarning)

api_key = os.environ['api_key']
co = cohere.Client(api_key)

client = cohere.Client(api_key)


class MidwitGPT:

  def __init__(self, client):
    self.client = client

  def generate(self, prompt, max_length=12, temperature=0.5):
    response = self.client.generate(prompt=prompt,
                                    temperature=temperature,
                                    max_tokens=max_length,
                                    stop_sequences=["--"])
    return response.generations[0].text


midwit_gpt = MidwitGPT(client)


def strip_prefix(view):
  index = view.find(':')
  if index != -1:
    stripped_view = view[index + 1:].strip()
    return stripped_view
  else:
    return view
    
def check_and_strip_incomplete_sentence(view):
  sentence_endings = ['.', '!', '?', ';']
  if view[-1] in sentence_endings:
      return view
  else:
      last_punctuation_index = max(view.rfind(char) for char in sentence_endings)
      if last_punctuation_index != -1:
          stripped_view = view[:last_punctuation_index + 1]
          return stripped_view.strip()
      else:
          return view
        
def process_text(input_text, word_limit, line_limit):
  latin1_encoded_text = input_text.encode('latin-1', errors='ignore')
  latin1_decoded_text = latin1_encoded_text.decode('latin-1')
  processed_text = latin1_decoded_text
  words = processed_text.split()
  lines = [
      ' '.join(words[i:i + word_limit])
      for i in range(0, len(words), word_limit)
  ]
  return '\n'.join(lines[:line_limit])


def generate_views(user_input):
  prompt = user_input
  simplistic_prompt = generate_simplistic_prompt(prompt)
  midwit_prompt = generate_midwit_prompt(prompt)
  simplistic_view = midwit_gpt.generate(simplistic_prompt,
                                        temperature=0.8,
                                        max_length=20)
  simplistic_view = strip_prefix(simplistic_view)
  simplistic_view = check_and_strip_incomplete_sentence(simplistic_view)
  midwit_view = midwit_gpt.generate(midwit_prompt,
                                    max_length=100,
                                    temperature=0.5)
  midwit_view = strip_prefix(midwit_view)
  midwit_view = check_and_strip_incomplete_sentence(midwit_view)

  return simplistic_view, midwit_view


def draw_lines(draw, lines, font, font_color, position):
  x, y = position
  draw.multiline_text((x, y),
                 lines,
                 fill=font_color,
                 font=font)


def generate_meme_image(prompt):
  simplistic_view, midwit_view = generate_views(prompt)
  texts = {"Simplistic view": simplistic_view, "Midwit view": midwit_view}
  midwit_lines = process_text(texts["Midwit view"],
                              word_limit=10,
                              line_limit=None)
  simplistic_lines = process_text(texts["Simplistic view"],
                                  word_limit=4,
                                  line_limit=None)

  font_color = (0, 0, 0)
  font_size = 22
  font = ImageFont.truetype("arial.ttf", font_size)
  positions = {
      "left_character": (240, 700),
      "midwit_character": (400, 240),
      "right_character": (1000, 620)
  }

  img = Image.open("bell-curve-blank-padded.jpg")
  draw = ImageDraw.Draw(img)

  draw_lines(draw, midwit_lines, font, font_color,
             positions["midwit_character"])
  draw_lines(draw, simplistic_lines, font, font_color,
             positions["right_character"])
  draw_lines(draw, simplistic_lines, font, font_color,
             positions["left_character"])

  return img
