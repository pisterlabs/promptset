import os
import sys
import random
import re
import ast
from rapidfuzz.distance import Levenshtein
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..", ".."))
from src.utils import utils

# Path to the OpenAI responses
input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..", "..", "src", "resources", "open_ai_results")
# Path where to save the final dictionary of all the alternatives, for each javadoc tag
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..", "..", "src", "resources", "data_augmentation")
# Import list of patterns to apply to improve variety of javagoc tags
_, patterns = utils.import_json(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..", "..", "src", "resources", "data_augmentation_patterns.json"))
# Import list of sets of original javadoc tags
_,original_javadoctags_set = utils.import_json(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..", "..", "src", "resources", "original_javadoctags_set.json"))
# Auxiliary variables for statistics and analysis purposes
matched_list = []
not_matched_list = []
# Instantiate dictionary of all the alternatives, for each javadoc tag
output_dict = {}


def apply_javadoc_patterns(javadoc_tags, patterns):
  new_alternatives = []
  # Iterate over the list of javadoc tags
  for javadoc_tag in javadoc_tags:
    # Iterate over the list of patterns (each pattern is a list composed of words with equivalent meaning)
    for word_list in patterns:
      # Iterate over each word of the current pattern list
      for word in word_list:
        # Randomly apply changes
        alternative = javadoc_tag.replace(word, random.choice(word_list))
    # Remove exceeding white spaces
    alternative = alternative.strip()
    # Apply other heuristics on punctuation
    if alternative.endswith('.'):
      # Randomly remove "." at the end of a javadoc tag
      if random.random() < 0.4:
        alternative = alternative[:-1]
    else:
      # Randomly add "." at the end of a javadoc tag
      if random.random() < 0.4:
        alternative = alternative + "."
    new_alternatives.append(alternative)
  # Always add empty javadoc tags to the list of the alternatives
  new_alternatives.append("")
  # Return the final list of new alternatives
  return new_alternatives


def match_original_javadoc_tag(javadoc_tag, eligibles):
  # Initialize best distance to +inf
  best_distance = float('inf')
  # The variable keep note of the current best matching
  most_probable_javadoc_tag = None
  # Iterate over the eligible javadoc tags
  for eligible in eligibles:
    # Compute the Levenshtein distance among the javadoc tag reported by OpenAI and the current eligible original one
    distance = Levenshtein.distance(javadoc_tag, eligible)
    # Check if the value of the distance is lower than the current best one
    if distance < best_distance:
      # Update best distance and matching javadoc tag
      best_distance = distance
      most_probable_javadoc_tag = eligible
  # Return the matched original javadoc tag
  return most_probable_javadoc_tag


if __name__ == '__main__':
  for idx, openai_filename in enumerate(os.listdir(input_path)):
    print(f"Processing: {openai_filename}")
    _, openai_output_str = utils.import_json(os.path.join(input_path, openai_filename))
    idx_openai_output = int(openai_filename.replace("output_","").replace(".json",""))
    original_set = original_javadoctags_set[idx_openai_output]
    # Pattern for matching content between "[\n  {\n" and " ]\n  }\n]"
    pattern = r"\[\s*{([\s\S]*?)]\s*}\s*]"
    # Find the first match using the pattern
    match = re.search(pattern, openai_output_str)

    if match:
      print("Match found! Generating alternative javadoc tags from OpenAI output...")
      # Extract the matched content applying regex expression to fix OpenAI output and extract json from string
      matched_content = match.group(0)
      matched_content = matched_content.replace("'tag'", "\"tag\"")
      matched_content = matched_content.replace("'alternatives'", "\"alternatives\"")
      matched_content = matched_content.replace("\\\\n", " ")
      matched_content = matched_content.replace("\\\\r", " ")
      matched_content = matched_content.replace("\\\\t", " ")
      matched_content = matched_content.replace("\\\\v", " ")
      matched_content = matched_content.replace("\\n", " ")
      matched_content = matched_content.replace("\\r", " ")
      matched_content = matched_content.replace("\\t", " ")
      matched_content = matched_content.replace("\\v", " ")
      matched_content = re.sub(r"\s+", " ", matched_content.strip())
      matched_content = re.sub(r"'@", "\"@", matched_content)
      matched_content = re.sub(r"',\s*\"@", "\", \"@", matched_content)
      matched_content = re.sub(r"',\s*\"alternatives\"", "\", \"alternatives\"", matched_content)
      matched_content = re.sub(r"\"((?!((\\*n|\\*r|\\*v|\\*t|\\*f)*\s*@))(?!((\\*n|\\*r|\\*v|\\*t|\\*f)*(\\*n|\\*r|\\*v|\\*t|\\*f)*\s*]))(?!((\\*n|\\*r|\\*v|\\*t|\\*f)*\s*,(\\*n|\\*r|\\*v|\\*t|\\*f)*\s*\"@))(?!((\\*n|\\*r|\\*v|\\*t|\\*f)*\s*,(\\*n|\\*r|\\*v|\\*t|\\*f)*\s*\"alternatives))(?!alternatives)(?!(\\*n|\\*r|\\*v|\\*t|\\*f)*\s*:(\\*n|\\*r|\\*v|\\*t|\\*f)*\s*\"@)(?!(\\*n|\\*r|\\*v|\\*t|\\*f)*\s*:(\\*n|\\*r|\\*v|\\*t|\\*f)*\s*\[)(?!tag))", "'",matched_content)
      matched_content = re.sub(r"'(\\*n|\\*r|\\*v|\\*t|\\*f)*\s*,(\\*n|\\*r|\\*v|\\*t|\\*f)*\s*]", "\" ]",matched_content)
      matched_content = re.sub(r"'(\\*n|\\*r|\\*v|\\*t|\\*f)*\s*,(\\*n|\\*r|\\*v|\\*t|\\*f)*\s*\"@", "\"",matched_content)
      matched_content = re.sub(r"'(\\*n|\\*r|\\*v|\\*t|\\*f)*\s*]", "\" ]",matched_content)
      matched_content = re.sub(r"'(\\*n|\\*r|\\*v|\\*t|\\*f)*\s*\"@", "\", \"@",matched_content)
      # Exctract json from processed OpenAI output string
      output_json = ast.literal_eval(matched_content)
      # Iterate over each javadoc tag
      for javadoc_tag_obj in output_json:
        # Apply patterns to generate variegate alternatives
        javadoc_tag_obj["alternatives"] = apply_javadoc_patterns(javadoc_tag_obj["alternatives"], patterns)
        # Match original javadoc tag with the one reported within the OpenAI output (it could be different due to OpenAI nondeterminism)
        key = match_original_javadoc_tag(javadoc_tag_obj["tag"], original_set)
        # Add javadoc tags alternatives to output dict of all the javadoc tags
        output_dict[key] = javadoc_tag_obj["alternatives"]
      matched_list.append(idx_openai_output)
    else:
      print("Match not found! OpenAI output not recognized...")
      not_matched_list.append(idx_openai_output)

  # Save output dict of all the javadoc tags
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  utils.export_stats(os.path.join(output_path, f"output_dict.json"), output_dict)

  # Log statistics
  print(f"OpenAI responses processed: {len(matched_list) + len(not_matched_list)}")
  print(f"OpenAI responses matched counter: {len(matched_list)}")
  print(f"OpenAI responses not matched counter: {len(not_matched_list)}")
  print(f"OpenAI responses not matched list: {sorted(not_matched_list)}")