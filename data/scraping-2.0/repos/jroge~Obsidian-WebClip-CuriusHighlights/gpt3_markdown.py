#cmd-line linter proof of concept for small bug where text is't paragraph formatted
#need a big context area though and feel like i would rather do it by hand then spend like tens of cents a blog
# import openai
# import os
# import yaml
# import requests

# # Set your OpenAI API key
# openai.api_key = ''

# def get_file_content(file_path):
#     with open(file_path, 'r') as file:
#         return file.read()

# def update_markdown(file_path, new_content):
#     with open(file_path, 'w') as file:
#         file.write(new_content)

# def extract_yaml_and_body(content):
#     if content.startswith('---'):
#         parts = content.split('---')
#         return parts[1], '---'.join(parts[2:])
#     return '', content

# def gpt3_response(prompt):
#     headers = {
#         'Content-Type': 'application/json',
#         'Authorization': f'Bearer {openai.api_key}'
#     }
#     data = {
#         "model": "gpt-3.5-turbo",
#         "messages": [{"role": "user", "content": prompt}],
#     }
#     response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
#     response_json = response.json()
    
#     # Check for errors or unexpected response structure
#     if 'choices' not in response_json or not response_json['choices']:
#         print("Error or unexpected response:", response.text)
#         return None  # or handle this situation as you see fit
    
#     # Assuming the first choice's message's content is what we want
#     return response_json['choices'][0]['message']['content'].strip()

# def main():
#     file_name = input("Enter the name of the markdown file (without .md): ")
#     file_path = '/Users/jamesrogers/Library/Mobile Documents/iCloud~md~obsidian/Documents/kepano-obsidian-main/Clippings/{}.md'.format(file_name)

#     if not os.path.exists(file_path):
#         print("File does not exist. Please check the file name.")
#         return

#     content = get_file_content(file_path)
#     yaml_header, body = extract_yaml_and_body(content)
#     body += " format this into paragraphs keep the syntax like == == but make sure the words with == == around it are in sentences and paragraphs without extraneous whitespace " + body
#     response = gpt3_response(body)
#     print(body)

#     # Combine the original YAML header with the new content
#     updated_content = f"---{yaml_header}---\n\n{response}"
#     update_markdown(file_path, updated_content)
#     print(f"Updated {file_name}.md with GPT-3's response.")

# if __name__ == '__main__':
#     main()
