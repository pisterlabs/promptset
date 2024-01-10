# !pip install openai --quiet
# !pip install EbookLib --quiet

import openai
import os
from ebooklib import epub
import base64
import os
import requests


openai.api_key = "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"       # get it at https://platform.openai.com/
stability_api_key = "sk-ATRFzLsqx4GmE1Yr3ddQGNa1ay8vHUl3TOO4auxRw9ktmwgm" # get it at https://beta.dreamstudio.ai/


# Using Google Translate for translations
# !pip install googletrans==3.1.0a0 --quiet

from googletrans import Translator
from googletrans.constants import LANGUAGES


# To find the language code to use in the cell form above.
for key, value in LANGUAGES.items():
    print(f"Code: {key}\t Language: {value}")

    #@title #Settings
#@markdown Program parameters
myModel = "gpt-4-0613"  #@param ['gpt-3.5-turbo-16k', 'gpt-4-0613'] {allow-input: true}
num_chapters = 7  #@param {type: "slider", min: 1, max: 30}
novelStyle = 'fantasy'   #@param {type: "string"}
author = "GPT-Author" #@param {type: "string"}
prompt = 'Alex lives in Paris in 2050, where the effects of global warming are making it difficult to find food and water.'  #@param {type: "string"}
writing_style = "Clear and easily understandable, similar to a young adult novel. Highly descriptive and sometimes long-winded. Similar to the pulse-pounding intensity of J. R. R. Tolkien, or Stephen King or Agatha Cristie."  #@param {type: "string"}
destLanguage = "en"  #@param {type: "string"}
#@markdown ---

def translate(text, dest='en', src='auto'):
    translator = Translator()
    try:
        txt = translator.translate(text, dest=dest, src=src).text
        #print(txt)
        return txt
    except:
        return text
    

def generate_cover_prompt(plot):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": translate("You are a creative assistant that writes a spec for the cover art of a book, based on the book's plot.", dest=destLanguage)},
            {"role": "user", "content": f"Plot: {plot}\n\n--\n\nDescribe the cover we should create, based on the plot. This should be two sentences long, maximum."}
        ]
    )
    return response['choices'][0]['message']['content']


def create_cover_image(plot):

  plot = str(generate_cover_prompt(plot))

  engine_id = "stable-diffusion-xl-beta-v2-2-2"
  api_host = os.getenv('API_HOST', 'https://api.stability.ai')
  api_key = stability_api_key

  if api_key is None:
      raise Exception("Missing Stability API key.")

  response = requests.post(
      f"{api_host}/v1/generation/{engine_id}/text-to-image",
      headers={
          "Content-Type": "application/json",
          "Accept": "application/json",
          "Authorization": f"Bearer {api_key}"
      },
      json={
          "text_prompts": [
              {
                  "text": plot
              }
          ],
          "cfg_scale": 7,
          "clip_guidance_preset": "FAST_BLUE",
          "height": 768,
          "width": 512,
          "samples": 1,
          "steps": 30,
      },
  )

  if response.status_code != 200:
      raise Exception("Non-200 response: " + str(response.text))

  data = response.json()

  for i, image in enumerate(data["artifacts"]):
      with open(f"cover.png", "wb") as f: # replace this if running locally, to where you store the cover file
          f.write(base64.b64decode(image["base64"]))


def create_epub(title, author, chapters, cover_image_path='cover.png'):
    book = epub.EpubBook()

    # Set metadata
    book.set_identifier('id123456')
    book.set_title(title)
    #book.set_language('en')
    book.set_language(destLanguage)
    book.add_author(author)

    # Add cover image
    with open(cover_image_path, 'rb') as cover_file:
        cover_image = cover_file.read()
    book.set_cover('cover.png', cover_image)

    # Create chapters and add them to the book
    epub_chapters = []
    for i, chapter_dict in enumerate(chapters):
        full_chapter_title = list(chapter_dict.keys())[0]
        chapter_content = list(chapter_dict.values())[0]
        if ' - ' in full_chapter_title:
            chapter_title = full_chapter_title.split(' - ')[1]
        else:
            chapter_title = full_chapter_title

        chapter_file_name = f'chapter_{i+1}.xhtml'
        epub_chapter = epub.EpubHtml(title=chapter_title, file_name=chapter_file_name, lang='en')

        # Add paragraph breaks
        formatted_content = ''.join(f'<p>{paragraph.strip()}</p>' for paragraph in chapter_content.split('\n') if paragraph.strip())

        epub_chapter.content = f'<h1>{chapter_title}</h1>{formatted_content}'
        book.add_item(epub_chapter)
        epub_chapters.append(epub_chapter)


    # Define Table of Contents
    book.toc = (epub_chapters)

    # Add default NCX and Nav files
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    # Define CSS style
    style = '''
    @namespace epub "http://www.idpf.org/2007/ops";
    body {
        font-family: Cambria, Liberation Serif, serif;
    }
    h1 {
        text-align: left;
        text-transform: uppercase;
        font-weight: 200;
    }
    '''

    # Add CSS file
    nav_css = epub.EpubItem(uid="style_nav", file_name="style/nav.css", media_type="text/css", content=style)
    book.add_item(nav_css)

    # Create spine
    book.spine = ['nav'] + epub_chapters

    # Save the EPUB file
    epub.write_epub(f'{title}.epub', book)



import openai
import random
import json
import ast

def print_step_costs(response, model):
  input = response['usage']['prompt_tokens']
  output = response['usage']['completion_tokens']

  if model == "gpt-4" or model == "gpt-4-0613":
    input_per_token = 0.00003
    output_per_token = 0.00006
  if model == "gpt-3.5-turbo-16k":
    input_per_token = 0.000003
    output_per_token = 0.000004
  if model == "gpt-4-32k-0613" or model == "gpt-4-32k":
    input_per_token = 0.00006
    output_per_token = 0.00012
  if model == "gpt-3.5-turbo" or model == "gpt-3.5-turbo-0613":
    input_per_token = 0.0000015
    output_per_token = 0.000002

  input_cost = int(input) * input_per_token
  output_cost = int(output) * output_per_token

  total_cost = input_cost + output_cost
  print('step cost:', total_cost)

def generate_plots(prompt):
    response = openai.ChatCompletion.create(
        model=myModel, #model="gpt-4-0613",
        messages=[
            {"role": "system", "content": translate(f"You're a creative assistant who generates captivating plots for {novelStyle} novels.", dest=destLanguage)},
            {"role": "user", "content": translate(f"Generate 10 {novelStyle} novel plots based on this prompt: {prompt}", dest=destLanguage)}
        ]
    )
    print_step_costs(response, myModel) #print_step_costs(response, "gpt-4-0613")
    return response['choices'][0]['message']['content'].split('\n')

def select_most_engaging(plots):
    response = openai.ChatCompletion.create(
        model=myModel, #model="gpt-4-0613",
        messages=[
            {"role": "system", "content": translate(f"You're an expert in writing {novelStyle} novel plots.", dest=destLanguage)},
            {"role": "user", "content": translate(f"Here are a number of possible plots for a new novel: {plots}\n\n--\n\nNow, write the final plot that we will go with. It can be one of these, a mix of the best elements of multiple, or something completely new and better. The most important thing is the plot should be fantastic, unique, and engaging.", dest=destLanguage)}
        ]
    )
    print_step_costs(response, myModel) #print_step_costs(response, "gpt-4-0613")
    return response['choices'][0]['message']['content']

def improve_plot(plot):
    response = openai.ChatCompletion.create(
        model=myModel, #model="gpt-4-0613",
        messages=[
            {"role": "system", "content": translate("You are an expert in improving and refining story plots.", dest=destLanguage)},
            {"role": "user", "content": translate(f"Improve this plot: {plot}", dest=destLanguage)}
        ]
    )
    print_step_costs(response, myModel) #print_step_costs(response, "gpt-4-0613")
    return response['choices'][0]['message']['content']

def get_title(plot):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": translate("You are an expert writer.", dest=destLanguage)},
            {"role": "user", "content": translate(f"Here is the plot: {plot}\n\nWhat is the title of this book? Just respond with the title, do nothing else.", dest=destLanguage)}
        ]
    )
    print_step_costs(response, "gpt-3.5-turbo-16k")
    return response['choices'][0]['message']['content']

def write_first_chapter(plot, first_chapter_title, writing_style):
    response = openai.ChatCompletion.create(
        model=myModel, #model="gpt-4-0613",
        messages=[
            {"role": "system", "content": translate(f"You're a world-class {novelStyle} writer.", dest=destLanguage)},
            {"role": "user", "content": translate(f"Here is the high-level plot to follow: {plot}\n\nWrite the first chapter of this novel: `{first_chapter_title}`.\n\nMake it incredibly unique, engaging, and well-written.\n\nHere is a description of the writing style you should use: `{writing_style}`\n\nInclude only the chapter text. There is no need to rewrite the chapter name.", dest=destLanguage)}
        ]
    )
    print_step_costs(response, myModel) #print_step_costs(response, "gpt-4-0613")

    improved_response = openai.ChatCompletion.create(
        model=myModel, #model="gpt-4-32k-0613",
        messages=[
            {"role": "system", "content": translate(f"You're a world-class {novelStyle} writer. Your job is to take your student's rough initial draft of the first chapter of their {novelStyle} novel, and rewrite it to be significantly better, with much more detail.", dest=destLanguage)},
            {"role": "user", "content": translate(f"Here is the high-level plot you asked your student to follow: {plot}\n\nHere is the first chapter they wrote: {response['choices'][0]['message']['content']}\n\nNow, rewrite the first chapter of this novel, in a way that is far superior to your student's chapter. It should still follow the exact same plot, but it should be far more detailed, much longer, and more engaging. Here is a description of the writing style you should use: `{writing_style}`", dest=destLanguage)}
        ]
    )
    print_step_costs(response, myModel) #print_step_costs(response, "gpt-4-32k-0613")
    return improved_response['choices'][0]['message']['content']

def write_chapter(previous_chapters, plot, chapter_title):
    try:
        i = random.randint(1,2242)
        # write_to_file(f'write_chapter_{i}', f"Plot: {plot}, Previous Chapters: {previous_chapters}\n\n--\n\nWrite the next chapter of this novel, following the plot and taking in the previous chapters as context. Here is the plan for this chapter: {chapter_title}\n\nWrite it beautifully. Include only the chapter text. There is no need to rewrite the chapter name.")
        response = openai.ChatCompletion.create(
            model=myModel, #model="gpt-4-0613",
            messages=[
                {"role": "system", "content": translate(f"You're a world-class {novelStyle} writer.", dest=destLanguage)},
                {"role": "user", "content": translate(f"Plot: {plot}, Previous Chapters: {previous_chapters}\n\n--\n\nWrite the next chapter of this novel, following the plot and taking in the previous chapters as context. Here is the plan for this chapter: {chapter_title}\n\nWrite it beautifully. Include only the chapter text. There is no need to rewrite the chapter name.", dest=destLanguage)}
            ]
        )
        print_step_costs(response, myModel) #print_step_costs(response, "gpt-4-0613")
        return response['choices'][0]['message']['content']
    except:
        response = openai.ChatCompletion.create(
            model=myModel, #model="gpt-4-32k-0613",
            messages=[
                {"role": "system", "content": translate(f"You're a world-class {novelStyle} writer.", dest=destLanguage)},
                {"role": "user", "content": translate(f"Plot: {plot}, Previous Chapters: {previous_chapters}\n\n--\n\nWrite the next chapter of this novel, following the plot and taking in the previous chapters as context. Here is the plan for this chapter: {chapter_title}\n\nWrite it beautifully. Include only the chapter text. There is no need to rewrite the chapter name.", dest=destLanguage)}
            ]
        )
        print_step_costs(response, myModel) #print_step_costs(response, "gpt-4-32k-0613")
        return response['choices'][0]['message']['content']


def generate_storyline(prompt, num_chapters):
    print("Generating storyline with chapters and high-level details...")
    json_format = """[{"Chapter CHAPTER_NUMBER_HERE - CHAPTER_TITLE_GOES_HERE": "CHAPTER_OVERVIEW_AND_DETAILS_GOES_HERE"}, ...]"""
    response = openai.ChatCompletion.create(
        model=myModel, #model="gpt-4-0613",
        messages=[
            {"role": "system", "content": translate(f"You are a world-class {novelStyle} writer. Your job is to write a detailed storyline, complete with chapters, for a {novelStyle} novel. Don't be flowery -- you want to get the message across in as few words as possible. But those words should contain lots of information.", dest=destLanguage)},
            {"role": "user", "content": translate(f'Write a fantastic storyline with {num_chapters} chapters and high-level details based on this plot: {prompt}.\n\nDo it in this list of dictionaries format {json_format}', dest=destLanguage)}
        ]
    )
    print_step_costs(response, myModel) #print_step_costs(response, "gpt-4-0613")

    improved_response = openai.ChatCompletion.create(
        model=myModel, #model="gpt-4-0613",
        messages=[
            {"role": "system", "content": translate(f"You are a world-class {novelStyle} writer. Your job is to take your student's rough initial draft of the storyline of a {novelStyle} novel, and rewrite it to be significantly better.", dest=destLanguage)},
            {"role": "user", "content": translate(f"Here is the draft storyline they wrote: {response['choices'][0]['message']['content']}\n\nNow, rewrite the storyline, in a way that is far superior to your student's version. It should have the same number of chapters, but it should be much improved in as many ways as possible. Remember to do it in this list of dictionaries format {json_format}", dest=destLanguage)}
        ]
    )
    print_step_costs(improved_response, myModel) #print_step_costs(improved_response, "gpt-4-0613")
    return improved_response['choices'][0]['message']['content']


def write_to_file(prompt, content):

    # Create a directory for the prompts if it doesn't exist
    if not os.path.exists('prompts'):
        os.mkdir('prompts')

    # Replace invalid characters for filenames
    valid_filename = ''.join(c for c in prompt if c.isalnum() or c in (' ', '.', '_')).rstrip()
    file_path = f'prompts/{valid_filename}.txt'

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f'Output for prompt "{prompt}" has been written to {file_path}\n')


def write_fantasy_novel(prompt, num_chapters, writing_style):
    plots = generate_plots(prompt)

    best_plot = select_most_engaging(plots)

    improved_plot = improve_plot(best_plot)


    title = get_title(improved_plot)

    storyline = generate_storyline(improved_plot, num_chapters)
    chapter_titles = ast.literal_eval(storyline)


    novel = f"Storyline:\n{storyline}\n\n"

    first_chapter = write_first_chapter(storyline, chapter_titles[0], writing_style.strip())
    novel += f"Chapter 1:\n{first_chapter}\n"
    chapters = [first_chapter]

    for i in range(num_chapters - 1):
        print(f"Writing chapter {i+2}...") # + 2 because the first chapter was already added
        chapter = write_chapter(novel, storyline, chapter_titles[i+1])
        novel += f"Chapter {i+2}:\n{chapter}\n"
        chapters.append(chapter)

    return novel, title, chapters, chapter_titles