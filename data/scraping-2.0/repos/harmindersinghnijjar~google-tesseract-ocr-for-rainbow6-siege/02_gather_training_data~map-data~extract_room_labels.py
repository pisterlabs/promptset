import csv
import os
import re
import time
import pandas as pd
import numpy as np
from regex import P
import openai

# Get OpenAI API key from environment variable
openai.api_key = os.environ.get('OPENAI_API_KEY')

# Method to ask GPT-3 the question extracted from Quora and return the respone text as a string.
def gpt3_completion(prompt, engine='text-davinci-003', temp=0.7, top_p=1.0, tokens=100, freq_pen=0.0, pres_pen=0.0):
        try:
            if not prompt:
                raise ValueError("Prompt cannot be empty.")
            prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen
            )
            if not response.choices:
                raise ValueError("No response from OpenAI API.")
            text = response.choices[0].text.strip()
            if not text:
                raise ValueError("Response text cannot be empty.")
            return text
        except Exception as e:
            print(f"Error occurred: {e}")
            return None

directory_path = os.getcwd()

total_room_labels = 0

for filename in os.listdir(directory_path):
    if filename.endswith(".txt"):
        with open(os.path.join(directory_path, filename), 'r') as file:
            content = file.read()
            start_index = content.find('roomLabels: [')
            if start_index != -1:
                end_index = content.find(']', start_index)
                room_labels_section = content[start_index:end_index + 1]
                # Regex to extract only the floor and description values
                room_labels = re.findall(r'floor: [1-4],\s*.*description: [^,}]+', room_labels_section)
                print(room_labels)
                # Write room labels to file
                with open(os.path.join(directory_path, 'room_labels.txt'), 'a') as room_labels_file:
                    for room_label in room_labels:
                        room_labels_file.write(room_label + '\n')
                print(f"Found {len(room_labels)} room labels")
                total_room_labels += len(room_labels)
                
print(f"Total room labels: {total_room_labels}")
print('Done')
time.sleep(5)

room_labels_data = []
with open(os.path.join(directory_path, 'room_labels.txt'), 'r') as room_labels_file:
    for line in room_labels_file:
        floor = re.findall(r'floor: (-?\d+)', line)[0]
        top = re.findall(r'top: (-?\d+)', line)[0] if 'top:' in line else None
        left = re.findall(r'left: (-?\d+)', line)[0] if 'left:' in line else None
        hardToRead = re.findall(r'hardToRead: (true|false)', line)[0] if 'hardToRead:' in line else None
        description = re.findall(r'description: (.+)', line)[0]
        room_labels_data.append([floor, top, left, hardToRead, description])

room_labels_df = pd.DataFrame(room_labels_data, columns=['floor', 'top', 'left', 'hardToRead', 'description']) # Create dataframe
room_labels_df['description'] = room_labels_df['description'].apply(lambda x: x[x.find('.') + 1:]) # Remove the prefix
room_labels_df['description'] = room_labels_df['description'].apply(lambda x: x.replace('removeBreakTags()', ' ')) # Remove removeBreakTags()
room_labels_df = room_labels_df.drop(['top', 'left', 'hardToRead'], axis=1) # Drop unnecessary columns
room_labels_df = room_labels_df.drop_duplicates(subset=['description']) # Drop duplicate descriptions

room_labels_df['floor'] = room_labels_df['floor'].apply(lambda x: x + 'F' if x in ['1', '2', '3', '4'] else x) # Add F to floor number
room_labels_df['floor'] = room_labels_df['floor'].apply(lambda x: 'B' if x in ['-1', '-2', '-3', '-4'] else x) # Replace -1, -2, -3, -4 with B
room_labels_df['floor'] = room_labels_df['floor'].apply(lambda x: 'G' if x == '0' else x) # Replace 0 with G

print(room_labels_df.head()) # Print first 5 rows

# Save to csv
room_labels_df.to_csv(os.path.join(directory_path, 'room_labels.csv'), index=False)
print('Done')
time.sleep(5)

# Loop through each row in csv and generate a prompt for each row so that we have locations with correct punctuation and acronyms
with open(os.path.join(directory_path, 'room_labels.csv'), 'r') as room_labels_file:
    csv_reader = csv.reader(room_labels_file)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        floor = row[0]
        description = row[1]
        print(f'Floor: {floor} Location: {description}')
        prompt = f'\nFloor: {floor} Location: {description}. Rewrite as a class label for an object detection model ensure correct punctuation and acronyms.Remain concise and replace spaces with underscores.  If the description doesnt contain more than one uppercase letter, divide the description into words using the capital letters as the delimeter, other wise use the letter before the last upper case letter as the delimiter. Use lowercase letters throughout so that youre consistant across hundreds of labels. Keep the labels in the format (floor_#f_description) similar to floor_1f_location_garage_roof, floor_3f_location_high_roof, floor_3f_location_low_roof, floor_1f_location_archives"'
        print(prompt)
        response = gpt3_completion(prompt)  # Make sure to define this function elsewhere in your code
        if response:
            print(response)
            with open(os.path.join(directory_path, 'yolo_r6_location_labels.txt'), 'a') as yolo_r6_location_labels_file:
                yolo_r6_location_labels_file.write(response + '\n')
        else:
            print('Error')
        time.sleep(10)