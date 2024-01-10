import openai
import os
import re
import sys

openai.api_key = os.getenv("OPENAI_API_KEY")
is_debug = False
is_simulation = False

def extract_structs(filename):
    with open(filename, 'r') as file:
        content = file.read()

    regex = r'((typedef\s+)?struct\s+\w*\s*\{[^}]+\}\s*\w*;)'
    matches = re.findall(regex, content)
    structs = [match[0] for match in matches]

    return structs

def simplify_structs(structs):
  struct_names = []
  for struct in structs:
      match = re.match(r'(typedef\s+)?struct\s+(\w*)', struct)
      if match and match.group(2):
          name = match.group(2)
          struct_names.append(name)
  
  simplified_structs = []
  for struct in structs:
      lines = struct.split('\n')

      simplified_lines = []
      for line in lines:
        known_type = any(name in line for name in struct_names)
        not_func_ptr = not re.search(r'\(.*\)', line)
        bracket = '{' in line or '}' in line
        if (known_type and not_func_ptr) or bracket:
          simplified_lines.append(line)
          
      simplified_struct = '\n'.join(simplified_lines)
      simplified_structs.append(simplified_struct)
  return simplified_structs

def formatUml(content):
  if "@startuml" not in content:
    content = "@startuml\n" + content
  return content

def debug(msg):
  if is_debug:
    print(msg)

def usage():
  if is_debug:
    print(f"Usage: python3 {sys.argv[0]} SOURCE_FILE")
 
def main():
  if len(sys.argv) > 1:
    source_file = sys.argv[1]
  else:
    usage()
    sys.exit(1)

  structs = extract_structs(source_file)
  structs = simplify_structs(structs)

  prompt ="Code snippet:\n" + "\n".join(structs)
  prompt = prompt + "\nPlease generate one PlantUML class diagram to explain the relationship of these struct.\n\n@startuml\n"
  debug(f"\n\nprompt:\n\n{prompt}\n")

  messages = [
        {"role": "system",
         "content": "your are a intelligent software engineer"},]
  messages.append({"role": "user", "content": prompt},)

  if is_simulation:
    return

  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    temperature=0,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
  )

  uml = formatUml(response.choices[0].message.content.strip())
  print(uml)

if __name__ == "__main__":
  main()

