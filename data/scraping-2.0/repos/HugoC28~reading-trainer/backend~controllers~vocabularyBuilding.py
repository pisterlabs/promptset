from dotenv import load_dotenv
from openai import AzureOpenAI
import re
import json
from flask import jsonify
from utils import text_endpoint, dalle_endpoint, azure_api_key, marks, words
import requests

load_dotenv()

def parse_story_prompt(text):
  # Regular expressions to match the title, story parts, and prompts
  title_pattern = r"Title: \"([^\"]+)\""
  story_pattern = r"Story Part (\d+): \"([^\"]+)\""
  prompt_pattern = r"Prompt for DALLE \(Part (\d+)\): \"([^\"]+)\""

  # Extract title
  title_match = re.search(title_pattern, text)
  title = title_match.group(1) if title_match else None

  # Extract story parts and prompts
  stories = re.findall(story_pattern, text)
  prompts = re.findall(prompt_pattern, text)

  # Convert stories and prompts into a dictionary
  exercises = {}
  for story_part, story_text in stories:
    corresponding_prompt = next((prompt_text for part, prompt_text in prompts if part == story_part), None)
    exercises[story_part] = {"story": story_text, "prompt": corresponding_prompt}

  # Construct the final data structure
  data = { "Type":"Vocabulary Building", "Title": title, "Exercise": exercises}
  return data

#messages = [{"role":"system","content":"You are a reading exercise generator who is used to generate Vocabulary texts: They are texts with a controlled vocabulary, made in order for the patient to learn and remember certain words that are difficult to them. "}]

def generateVocabularyText(selected_topic, exercise_number, difficulty):

  messages = [{"role":"system","content":"You are a reading exercise generator who is used to generate Vocabulary texts: They are texts with a controlled vocabulary, made in order for the patient to learn and remember certain words that are difficult to them. "}]


  prompt = f'''Generate a reading exercise and a image prompt on the difficult words {selected_topic}. The exercise should consist of {exercise_number} parts, each with a controlled vocabulary suited for the {marks[difficulty-1]} level. Repeat the difficult words several time in the exercise. The text in each part should be approximately {words[difficulty-1]} words.\n\n For each part of the exercise, also provide a descriptive prompt for image generator to create an image that visually represents the story part.\n\n Format your response as follows:\n\n Title: "Title of the story"\nStory Part 1: "Generated story part 1"\n Prompt for DALLE (Part 1): "Image prompt describing story part 1"\n...\nStory Part {exercise_number}: "Generated story part {exercise_number}"\nPrompt for DALLE (Part {exercise_number}): "Image prompt describing story part {exercise_number}"'''

  messages.append({"role":"user","content":prompt})
  # Try to generate the exercise and prompts with gpt 4 in this try block.
  try:
    textClient = AzureOpenAI(
      api_version="2023-12-01-preview",  
      api_key=azure_api_key,  
      azure_endpoint=text_endpoint
    )

    print(messages)

    response = textClient.chat.completions.create(
      model="gpt-4", # model = "deployment_name".
      messages=messages
    )

    chatGPTReply = response.choices[0].message.content
    parsedText = parse_story_prompt(chatGPTReply)
  
  except requests.RequestException as e:
    print(f"Error in generating the exercise and prompts: {e}")
    return jsonify({"error": "Internal Server Error"}), 500


  # Try to generate the images in this try block.
  try:
    # Diffenrent models have different endpoints
    dalleClient = AzureOpenAI(
      api_version="2023-12-01-preview",  
      api_key=azure_api_key,  
      azure_endpoint=dalle_endpoint
    )

    # Loop through the prompts and sentences and generate the images
    for key, value in parsedText["Exercise"].items():
      
      result = dalleClient.images.generate(
        #model= "dall-e-3", # the name of your DALL-E 3 deployment
        prompt= value["prompt"]+"Use a cartoon style.",
        n=1
      )
      print(result)

      json_response = json.loads(result.model_dump_json())

      image_url = json_response["data"][0]["url"]  # extract image URL from response

      parsedText["Exercise"][key]["url"] = image_url

  except Exception as e:
    print(f"Error in generating the images: {e}")
    return jsonify({"error": "Internal Server Error"}), 500
    
  print("========================================\n")
  print("Parsed Text:")
  print(parsedText["Exercise"])
  return jsonify(parsedText), 200