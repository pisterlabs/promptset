from dotenv import load_dotenv
from openai import AzureOpenAI
import re
import json
from flask import jsonify
from utils import text_endpoint, dalle_endpoint, azure_api_key, words, marks
import requests

load_dotenv()

def parse_text_to_object(text):
  print(text)

  # Split the input string into parts using the "STORY:", "PROMPT:", "QUESTION:", and "ANSWERS:" markers
  parts = [part.strip() for part in text.split("STORY:")[1:]]

  # Extract title of the story
  fpart = text.split("STORY:")[0]
  title = fpart.split("TITLE:")[1]

  # Create a list of dictionaries
  result = {}
  for index, part in enumerate(parts):
    story, rest = part.split("PROMPT:")
    prompt, rest = rest.split("QUESTION:")
    question, rest = rest.split("TRUE_ANSWER:")
    true_answer, rest = rest.split("POSSIBLE_ANSWERS:")
    answers = [ans.replace(">","").strip() for ans in rest.strip().split('\n') if ans.strip()]


    # Construct the dictionary
    result_dict = {
      'story': story.strip(),
      'prompt': prompt.strip(),
      'question': question.strip(),
      'true_answer': true_answer.strip(),
      'answers': answers
    }

    # Append the dictionary to the result list
    result[index] = result_dict

  # For an standar structure on excersices, they will be a dictionary of two fields,
  # "Type", wich is obvious and "Exercise", which is the original content and "Title",
  # to give a title to the story for displaying purposes

  exercise = {
    "Type":"Reading Comprehension",
    "Exercise": result,
    "Title":title.strip()
  }

  return exercise



def generateComprehensionTest(selected_topic, nbr_parts, difficulty):

  messages  = [{"role":"system","content":"You are a reading exercise generator, adapted for a 9 years old child with language impairments."}]


  # The difficult words can be maybe asked from the user in the UI?
  prompt = f'''Compose a short and engaging story for a 9-year-old child with reading difficulties, centered around {selected_topic}. The story should be a {marks[difficulty-1]} level for a 9-year-old child. The sentences should be simple, with clear and consistent structure. Ensure that the text is cohesive and forms an engaging narrative about {selected_topic}, including aspects of their appearance, behavior, and environment. This story must contain {nbr_parts} parts, each part should be approximately {words[difficulty-1]} words. For each part, give on DALL-E prompts that describes the related part. Be consistent with the prompts and always describe the characters in the same way. Also add for each of those part one Multiple Choice Question of difficulty {marks[difficulty-1]} related to the part, to test the child's text comprehension. Try not to ask questions that can be answered only with the generated image, to really test child's text comprehension.\nYou must follow this exact structure, with i from 1 to {nbr_parts}, don't add any other details such as specific separators, part titles, transitions or advices :\nSTORY: <story's part i>\nPROMPT: <DALL-E script for part i>\nQUESTION: <MCQ question for part i>\nTRUE_ANSWER: <the true answer among the 4 possible answers>\nPOSSIBLE_ANSWERS: <4 possible answers for part i (containing TRUE_ANSWER, with the exact same syntax (letters and punctuation), at a random position, different for each question), separated by \n >\n Start the response with TITLE:<title of the story>'''

  messages.append({"role":"user","content":prompt})
  # Try to generate the exercise and prompts with gpt 4 in this try block.
  try:
    textClient = AzureOpenAI(
      api_version="2023-12-01-preview",  
      api_key=azure_api_key,  
      azure_endpoint=text_endpoint
    )

    response = textClient.chat.completions.create(
      model="gpt-4", # model = "deployment_name".
      messages=messages
    )

    chatGPTReply = response.choices[0].message.content
    parsedText = parse_text_to_object(chatGPTReply)
  
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
      print(key, value)

      result = dalleClient.images.generate(
        #model= "dall-e-3", # the name of your DALL-E 3 deployment
        prompt= value["prompt"]+"Use a cartoon style.",
        n=1
      )

      json_response = json.loads(result.model_dump_json())

      image_url = json_response["data"][0]["url"]  # extract image URL from response

      parsedText["Exercise"][key]["url"] = image_url

  except Exception as e:
    print(f"Error in generating the images: {e}")
    return jsonify({"error": "Internal Server Error"}), 500

  print(parsedText)
  return jsonify(parsedText), 200