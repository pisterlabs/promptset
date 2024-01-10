import os
import re

from openai_util import num_tokens_from_messages


def split_code(file_path, char_limit):
  """
     Splits the source code in a file into fragments of text not exceeding a
     specified character limit.

     Args:
     - file_path (str): The absolute path to the source code file.
     - char_limit (int): The maximum number of characters allowed in each fragment.

     Returns:
     - List[str]: A list of multiline strings representing the code fragments.

     Description:
     This function takes the absolute path to a source code file and a character
      limit as input. It then reads the contents of the file and splits it into
      fragments based on certain rules:

     1. Fragments are created by breaking the code at empty lines or lines with
     comments without indentation.
     2. Matching parentheses and indentation are counted to ensure fragments do
     not split in the middle of a method.
     3. The function is designed to work mostly correctly with PHP, Java, C++,
     and Python source code files.
  """
  with open(file_path, 'r', encoding='utf-8') as file:  # Ensure correct encoding
    code = file.read()

  fragments = []
  buffer = ""
  parenthesis_count = 0
  brace_count = 0
  bracket_count = 0
  php_tag_count = 0  # Counter for PHP tag sections
  in_block_comment = False
  in_import_block = False
  for line in code.splitlines(True):  # Keep line endings

    # Check if adding the new line would exceed char_limit
    if len(buffer) + len(line) > char_limit:
      fragments.append(buffer.strip())
      buffer = ""

    # Check for import statements block
    if re.match(r'^\s*(import|from|include|require|#include)\b', line):
      in_import_block = True
    elif in_import_block and not line.strip():
      in_import_block = False  # End of import block on first empty line
      continue  # Skip empty line at the end of import block

    # Skip processing if in import block
    if in_import_block:
      continue

    # Skip processing if inside a block comment
    if in_block_comment:
      buffer += line  # Add block comment line to buffer
      # Check if block comment ends on this line
      if '*/' in line:
        in_block_comment = False
      continue

    # Handle block comments
    if '/*' in line:
      in_block_comment = True
      buffer += line  # Ensure the opening line of block comment is included
      continue  # Skip further processing for this line

    # Count matching parentheses, braces, brackets, and PHP tags
    parenthesis_count += line.count('(') - line.count(')')
    brace_count += line.count('{') - line.count('}')
    bracket_count += line.count('[') - line.count(']')
    php_tag_count += line.count('<?php') - line.count('?>')

    # Check for comment or empty line with no indentation
    is_comment_or_empty = re.match(r'^\s*(//|#|--|/\*|\*/|\*)?\s*$', line)

    # Check conditions for splitting the fragment
    conditions = (
      parenthesis_count == 0,
      brace_count == 0,
      bracket_count == 0,
      php_tag_count == 0,
      is_comment_or_empty
    )
    if all(conditions):
      fragments.append(buffer.strip())
      buffer = ""
      # Reset counts
      parenthesis_count = 0
      brace_count = 0
      bracket_count = 0
      php_tag_count = 0  # Reset PHP tag counter
    else:
      buffer += line

  # Append any remaining code to the fragments
  if buffer:
    fragments.append(buffer.strip())

  return fragments


def split_code_to_optimal_fragments(at_path, target_file):
  """
  Find the maximum split size for a source code file to stay within a
  target token limit.

  This function iteratively adjusts the code fragment size until the generated
  code fragments
  do not exceed the specified target token limit. It aims to maximize the code
  fragment size
  while staying within the token limit.

  :param at_path: The path to the directory containing the target source code file.
  :param target_file: The name of the target source code file.
  :return: A list of code fragments that adhere to the target token limit.
  """
  start_code_fragment_size = 60000
  target_number_of_tokens = 25000
  abs_file_path = os.path.join(at_path, target_file)

  code_fragment_size = start_code_fragment_size

  code_fragments = None

  iteration_counter = 0
  while True:
    iteration_counter += 1

    if iteration_counter > 20:
      print("Too many iterations")  # Break out of loop if it takes too long to
      break

    new_code_fragments = split_code(abs_file_path,
                                                  code_fragment_size)
    max_code_fragment_tokens = max(num_tokens_from_messages(
      [{"role": "user", "content": code_fragment}])
                                   for code_fragment in new_code_fragments)
    if not code_fragments or max_code_fragment_tokens < target_number_of_tokens:
      code_fragments = new_code_fragments

    if len(code_fragments) == 1:
      # No need to split
      break

    # Try to increase the size of the code fragment
    if max_code_fragment_tokens < target_number_of_tokens and \
        (target_number_of_tokens-max_code_fragment_tokens)/target_number_of_tokens > 0.05:
      code_fragment_size += (code_fragment_size *
                             ((target_number_of_tokens-max_code_fragment_tokens) /
                              max_code_fragment_tokens) * 0.8)
    else:
      break

  return code_fragments