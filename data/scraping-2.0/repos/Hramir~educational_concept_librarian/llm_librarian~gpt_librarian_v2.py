import pandas as pd
import openai
import pickle
import sys
import json
import time
from logger import Logger
import math


verbose = False
debugging = False
logging = True
#model_id = "gpt-4-1106-preview"
model_id = "gpt-3.5-turbo-1106"
randomly_sample_subset = None # Set to None to process all transcripts, or to a number for just a random subsample (e.g., for testing)
video_transcripts_file = "video_transcripts_with_hierarchy_1702425161.csv"
concept_library_init_file = "concept_library_1702425161.pkl" # Set to None to build library from scratch


def print_dialog(prompt_1, response_1, video_json=None, concept_library=None):
  print("\n\n========== Prompt 1 ==================")
  print(prompt_1)

  print("\n\n=== Response 1 ===")
  print(response_1)

  if video_json is not None:
    print("\n\n========== Video JSON ================")
    print(video_json)

  if concept_library is not None:
    print("\n\n========== Concept library ===========")
    print(concept_library)


def standardize(concept):
  return concept.lower().replace("_", " ").replace(" = ", "=").replace(" x ", "x").replace(" + ", "+").replace("^", "")

def expand_concepts(json_data, concepts_set):
    """
    Function to expand the concepts set with unique concepts found in a nested JSON structure.

    :param json_data: List of nested JSON objects.
    :param concepts_set: Set of existing concepts.
    :return: Expanded set of concepts.
    """
    for item in json_data["lesson"]:
        # Add concepts from the current level to the set
        concepts_set.add(standardize(item["primary_concept"]))
        for concept in item["supporting_concepts"]:
            concepts_set.add(standardize(concept))
        
        # If there are nested activities, recurse into them
        if "activities" in item and item["activities"]:
            expand_concepts({"lesson": item["activities"]}, concepts_set)
    
    return concepts_set


def check_and_repair_json(json_obj):
  valid_activities = ["definition", "example", "visualization", "application", "proof", "analogy", "compare and contrast", "review", "other"]
  anomalies = []

  def create_empty_activity(original_activity_name="", upper_primary_concept=""):
    return {
      "activity": original_activity_name if isinstance(original_activity_name, str) else "",
      "primary_concept": upper_primary_concept if isinstance(upper_primary_concept, str) else "",
      "supporting_concepts": [],
      "activities": []
    }

  def check_activity(activity, path, valid_activities):
    # Reinitialize fields if they are not strings or lists as required
    if not isinstance(activity.get("activity", None), str):
      activity["activity"] = ""
      anomalies.append(f"Non-string 'activity' at {path}, reinitialized to empty string")

    if not isinstance(activity.get("primary_concept", None), str):
      activity["primary_concept"] = ""
      anomalies.append(f"Non-string 'primary_concept' at {path}, reinitialized to empty string")

    if not isinstance(activity.get("supporting_concepts", None), list):
      activity["supporting_concepts"] = []
      anomalies.append(f"Non-list 'supporting_concepts' at {path}, reinitialized to empty list")

    # Make sure it's a valid activity name
    if activity["activity"] not in valid_activities:
      anomalies.append(f"Non-standard activity: {activity['activity']}")

    # Check and fix 'activities' field
    if "activities" not in activity or not isinstance(activity["activities"], list):
      activity["activities"] = []
      anomalies.append(f"Missing or non-list 'activities' at {path}, initialized to empty list")
    else:
      for i, nested_activity in enumerate(activity["activities"]):
        if isinstance(nested_activity, str):
          # Replace string with an empty activity object
          activity["activities"][i] = create_empty_activity(nested_activity, activity["primary_concept"])
          anomalies.append(f"String 'activities' item replaced with empty activity at {path}[{i}]")
        elif not isinstance(nested_activity, dict):
          anomalies.append(f"Invalid 'activities' item at {path}[{i}], replaced with empty activity")
          activity["activities"][i] = create_empty_activity()
        else:
          check_activity(nested_activity, f"{path}[{i}]", valid_activities)

  # Clone the object to avoid modifying the original
  json_obj_clone = json.loads(json.dumps(json_obj))

  # Check the structure
  for i, lesson in enumerate(json_obj_clone["lesson"]):
    check_activity(lesson, f"lesson[{i}]", valid_activities)

  return json_obj_clone, anomalies


# Load the video transcripts
df = pd.read_csv(video_transcripts_file)

df = df.sort_values(by=['playlist_position'])

if randomly_sample_subset and randomly_sample_subset is not None: 
  df = df.sample(n=randomly_sample_subset, replace=False, random_state=randomly_sample_subset)

# New column for hierarchy
if "activity_concept_hierarchy" not in df.columns:
  df["activity_concept_hierarchy"] = ""

if concept_library_init_file is None:
  # Initialize the concept library set
  concept_library = set()
else:
  concept_library = set(pickle.load(open(concept_library_init_file, 'rb')))

print("INITIAL CONCEPT LIBRARY: ")
print(concept_library)

# Load prompt templates
with open("prompt_single.txt", "r") as file:
    prompt_1 = file.read()

start_timestamp = time.time()
if logging:
   sys.stdout = Logger("gpt_librarian_" + str(int(start_timestamp)) + ".log")

client = openai.OpenAI()

major_error_count = 0
json_structure_error_count = 0
prompt_token_count = 0
completion_token_count = 0
total_tries = 0

# Process each transcript in the dataframe
for index, row in df.iterrows():

  if pd.isna(df.at[index, "activity_concept_hierarchy"]) or df.at[index, "activity_concept_hierarchy"] == "":
    retries = 0
    success = False
    retry_limit = 3
    while retries < retry_limit and not success: 
      total_tries += 1
      if retries > 0: 
        print(f"RETRYING. Retried {retries} times")
      try:
        response_1_text = None
        prompt_1_filled = None
        video_json = None
        print("====================================================")
        print("============= Video ID: " + str(row["video_id"]) + " ================")

        # Replace <transcript> in prompt_1 with the actual transcript
        transcript = row["transcript"]
        prompt_1_filled = prompt_1.replace("<transcript>", transcript)

        # Send prompt_1 to OpenAI and get the response
        response_1 = client.chat.completions.create(
            model=model_id,  # Adjust the model as necessary
            response_format={"type": "json_object"},
            messages=[
              {"role": "system", "content": "You are are an expert data analyst. You are part of a research team studying the role of activity and concept hierarchies in determining the teaching quality of educational videos."},
              {"role": "user", "content": prompt_1_filled},
            ]
        )

        # Count tokens
        prompt_token_count = prompt_token_count + response_1.usage.prompt_tokens
        completion_token_count = completion_token_count + response_1.usage.completion_tokens

        response_1_text = response_1.choices[0].message.content.strip().lower()

        if response_1_text[0:7] == "```json": response_1_text = response_1_text[7:]
        if response_1_text[-3:] == "```": response_1_text = response_1_text[0:-3]
        video_json = json.loads(response_1_text)

        # Check and repair errors in json structure
        repaired_json, anomalies = check_and_repair_json(video_json)
        if len(anomalies) > 0:
          json_structure_error_count = json_structure_error_count + len(anomalies)
          print("WARNING: JSON structure errors encountered. Here is a summary of the errors:")
          print(anomalies)
          if retries < retry_limit - 1: # Retry unless we are on our last try
            raise ValueError("JSON structural error, retry")
          else:
            video_json = repaired_json

        # Save the JSON string to the dataframe
        df.at[index, "activity_concept_hierarchy"] = json.dumps(video_json)

        # Augment the concept library
        concept_library = expand_concepts(video_json, concept_library)

        if verbose: 
          print_dialog(prompt_1_filled, response_1_text)
        
        # Save the updated dataframe
        df.to_csv("video_transcripts_with_hierarchy_" + str(int(start_timestamp)) + ".csv", index=False)

        # Save the concept library using pickle
        with open("concept_library_" + str(int(start_timestamp)) + ".pkl", "wb") as file:
          pickle.dump(concept_library, file)

        success = True

      except Exception as e:
        major_error_count = major_error_count + 1
        print("ERROR. Printing dialog:")
        print_dialog(prompt_1_filled, response_1_text, video_json, concept_library)
        retries = retries + 1
        if debugging: 
          raise e
  else:
    print(f"Video of ID {str(row['video_id'])} already processed, skipping...")

print("FINAL CONCEPT LIBRARY:")
print(concept_library)

print("Total number of transcripts attempted:", len(df))
print("Major error count:", major_error_count)
print("JSON structure error count", json_structure_error_count)
print("Total tries:", total_tries)

print("RUNTIME:", time.time() - start_timestamp, "seconds")

print("Prompt token count:", prompt_token_count)
print("Completion token count:", completion_token_count)

print("PRICING:")
if model_id == "gpt-4-1106-preview":
  prompt_cost = prompt_token_count * ((0.01/1000))
  completion_cost = completion_token_count * ((0.03/1000))
elif model_id == "gpt-3.5-turbo-1106":
  prompt_cost = prompt_token_count * ((0.001/1000))
  completion_cost = completion_token_count * ((0.002/1000))
else: 
  print("Unspecified model type, unable to calculate cost.")
  prompt_cost = 0
  completion_cost = 0

print("Prompt cost: $" + str(prompt_cost))
print("Completion cost: $" + str(completion_cost))
print("TOTAL COST: $" + str(prompt_cost + completion_cost))

print("Saved as: " + "video_transcripts_with_hierarchy_" + str(int(start_timestamp)) + ".csv")