import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv() 
client = OpenAI()

# Specify the path to your JSON file
file_path = 'test_data.json'
data = []
passed = 0
failed = 0
responses = []
# Open and read the JSON file
with open(file_path, 'r', encoding='utf-8') as json_file:
    for line in json_file:
        # Process each line using json.loads()
        json_data = json.loads(line)

        # Now 'json_data' contains the data from the current line
        data.append(json_data)
# Now, 'data' contains the contents of your JSON file
# print(data[0])
for i in range(len(data)):
    response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[data[i]['messages'][0], data[i]['messages'][1]]
    )
    if response.choices[0].message.content == data[i]['messages'][2]['content']:
        passed += 1
    else:
        failed += 1
    print("Passed: ", passed, "Failed: ", failed, "Total: ", passed + failed)
    responses.append(response)

print("Passed: ", passed, "Failed: ", failed, "Total: ", passed + failed)

with open('test_gpt3_5_data_responses.json', 'w', encoding='utf-8') as json_file:
    for _ in responses:
        json_file.write(str(_))
        json_file.write("\n")

with open('test_gpt3_5_data_accuracy.txt', 'w', encoding='utf-8') as txt_file:
    txt_file.write("Passed: " + str(passed) + " Failed: " + str(failed) + " Total: " + str(passed + failed))
