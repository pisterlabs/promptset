# Extract attributes from the gpt4 response string
import traceback
import json
import openai
import os
from dotenv import load_dotenv
from tqdm import tqdm
import json
import time
from tqdm import tqdm
from openai import OpenAI, AsyncOpenAI
import json
from ratelimit import limits, sleep_and_retry
import asyncio
import os
import re
from aiolimiter import AsyncLimiter
load_dotenv()
openai.api_key = os.getenv('OPENAI_KEY')
api_key = os.getenv('OPENAI_KEY')  
client = AsyncOpenAI(api_key=api_key, timeout=30)
# Define the directory path where your JSON files are located
dir_path = '/home/ubuntu/RLHF/LLaVA-RLHF-Data'
output_path = '/home/ubuntu/RLHF/LLaVA-RLHF/RLHF/gpt4_ratings/training_data/teacher-critique/extracted_attributes_2308_3350_gpt3_5_async.json'
rate_limit_per_minute = 400
model = "gpt-3.5-turbo-1106"
rate_limit_per_day = 10000
# List all files in the directory and sort them numerically based on the filename
file_pattern = re.compile(r'llava_7b_v1_preference_subsample_gpt4v_response_(\d+)_(\d+).json')
files = os.listdir(dir_path)
json_files = sorted([file for file in files if file_pattern.match(file)], key=lambda x: int(file_pattern.match(x).group(1)))
json_files= ['/home/ubuntu/RLHF/LLaVA-RLHF-Data/llava_7b_v1_preference_subsample_gpt4v_response_3009_3109.json']
def _check_row(row):
    gpt_response = row['gpt4v_response']
    if isinstance(gpt_response, str):
        gpt_response = json.loads(gpt_response)
    assert isinstance(gpt_response, dict), "gpt_response is not a dict"
    if "error" in gpt_response:
        return False
    else:
        return True
    
def load_only_rows_wo_error(file):
    with open(file) as json_file:
        data = json.load(json_file)
        filtered_data = {key: row for key, row in data.items() if _check_row(row)}
        print("Total number of rows: ", len(data), 'for file: ', file.rsplit('/')[-1], 'after filtering: ', len(filtered_data))
        print("Ratio of rows kept: ", len(filtered_data)/len(data))
    return filtered_data

def load_all_files(file_list):
    data = {}
    for file in file_list:
        data.update(load_only_rows_wo_error(file))
    return data

print(f'Processing {json_files} files.')
# Initialize a list to store merged data
merged_data = load_all_files(json_files)


# json_path = '/home/ubuntu/RLHF/LLaVA-RLHF-Data/llava_7b_v1_preference_subsample_gpt4v_response_80_180.json'
# with open(json_path, 'r') as file:
#     list_of_dict = json.load(file)


system_message_json_extraction_version_2 = """
Givien a response that contain something like the following:
{'id': 'chatcmpl-8NzMDrSkcZpQRmw1mrRflUfxGZAhY', 'object': 'chat.completion', 'created': 1700728093, 'model': 'gpt-4-1106-vision-preview', 'usage': {'prompt_tokens': 2448, 'completion_tokens': 630, 'total_tokens': 3078}, 'choices': [{'message': {'role': 'assistant', 'content': "{'my_response': 'The purpose of a light rail transit (LRT) system is to offer an efficient, convenient, and sustainable mode of transportation within urban and suburban areas. It is designed to move people quickly and comfortably over short to medium distances, connecting different neighborhoods and key areas within a city. Light rail systems often run on dedicated tracks to avoid congestion and are powered by electricity, which helps reduce air pollution and carbon emissions. In addition to easing traffic congestion by serving as an alternative to car travel, LRT systems can promote economic development by enhancing accessibility to business districts and other important locations.',\n 'Ratings': {'Ratings4CandidateResponseA': {'CommentSection': 'Response A gives a detailed description of the purpose of an LRT system, outlining its role in urban transport, its design features, and its environmental benefits. The information is accurate, and the response provides a comprehensive overview of the LRT system, which seems beneficial for someone seeking an in-depth understanding.', 'Hallucinations': 0.0, 'Helpfulness': 0.9, 'Quality': 0.9, 'Spatial-Awareness': 0.0, 'Domain-Knowledge': 0.9},\n 'Ratings4CandidateResponseB': {'CommentSection': 'Response B explains the purpose of the LRT system with a focus on its operational characteristics, integration with other forms of transport, and its accessibility features. The response also mentions the benefits of the LRT system in terms of reducing traffic and environmental impact. However, it provides slightly less detail compared to Response A.', 'Hallucinations': 0.0, 'Helpfulness': 0.8, 'Quality': 0.85, 'Spatial-Awareness': 0.0, 'Domain-Knowledge': 0.8},\n 'Ratings4CandidateResponseC': {'CommentSection': 'Response C offers a comprehensive understanding of an LRT system\\'s purpose, highlighting its efficiency, environmental friendliness, and role in connecting urban and suburban areas. This response emphasizes the benefits, including mobility, reduced traffic, environmental advantages, and economic growth. It provides a similar level of detail as Response A and is accurate and informative.', 'Hallucinations': 0.0, 'Helpfulness': 0.9, 'Quality': 0.9, 'Spatial-Awareness': 0.0, 'Domain-Knowledge': 0.9},\n 'Ratings4YourOwnResponseYouWrote': {'CommentSection': 'My own response succinctly describes the purpose of an LRT system, emphasizing its efficiency, convenience, and sustainability. It touches on the main points like improved transportation and environmental advantages but does not dive as deeply into the technical aspects or the extended benefits such as economic growth. It remains accurate and adequately informative but is not as detailed as the candidate responses.', 'Hallucinations': 0.0, 'Helpfulness': 0.85, 'Quality': 0.85, 'Spatial-Awareness': 0.0, 'Domain-Knowledge': 0.8}}}"}, 'finish_details': {'type': 'stop', 'stop': '<|fim_suffix|>'}, 'index': 0}]}
Please extract the attibutes into the following json format with the example below:
{'my_response': 'The image shows a single airplane on the tarmac with the markings "Armavia" on its fuselage, suggesting it belongs to the now-defunct Armenian airline. There are no multiple airplanes visible in the picture. The aircraft appears to be an Airbus A319 or similar model based on the design features seen, such as the size, shape, and number of windows. There are service vehicles around the aircraft, including what appears to be a baggage conveyor and another utility vehicle, implying it is being serviced, perhaps for an upcoming flight or after landing. Despite the lack of multiple airplanes, the servicing activity suggests efficient ground operations, necessary for quick turnaround times and maintaining flight schedules.',
 'Ratings': {'Ratings4CandidateResponseA': {'CommentSection': 'Response A incorrectly assumes multiple airplanes are present and elaborates on implications that are not relevant to the actual image. The description of large airplanes and implications for international flights, while potentially accurate in a different context, does not apply here.',
   'Hallucinations': 1.0,
   'Helpfulness': 0.1,
   'Quality': 0.1,
   'Spatial-Awareness': 0.0,
   'Domain-Knowledge': 0.1},
  'Ratings4CandidateResponseB': {'CommentSection': 'Response B makes a similar error as Response A, discussing multiple airplanes when only one is present. Though the general implications it discusses could be accurate in a different situation, it is not relevant to this image.',
   'Hallucinations': 1.0,
   'Helpfulness': 0.1,
   'Quality': 0.1,
   'Spatial-Awareness': 0.0,
   'Domain-Knowledge': 0.1},
  'Ratings4CandidateResponseC': {'CommentSection': 'Response C also hallucinates the presence of multiple airplanes when only one airplane is visible, making the implications mentioned irrelevant. The response does identify the need for efficient management and customer experience, which are correct in general airport scenarios, but it still fails to address what is actually depicted.',
   'Hallucinations': 1.0,
   'Helpfulness': 0.1,
   'Quality': 0.1,
   'Spatial-Awareness': 0.0,
   'Domain-Knowledge': 0.1},
  'Ratings4YourOwnResponseYouWrote': {'CommentSection': 'My response avoids the hallucination present in the other candidates by accurately describing the single airplane that is visible in the image. It also correctly identifies the type of aircraft and the operational activities taking place, such as servicing, without making incorrect assumptions about the number of airplanes or the implications of such a scene.',
   'Hallucinations': 0.0,
   'Helpfulness': 0.9,
   'Quality': 0.9,
   'Spatial-Awareness': 1.0,
   'Domain-Knowledge': 0.8}}}
Note that the main keys are always ['Ratings4CandidateResponseA'], ['Ratings4CandidateResponseB'], ['Ratings4CandidateResponseC'], ['Ratings4YourOwnResponseYouWrote'] and the
subkeys are always {'CommentSection'}, {'Hallucinations'}, {'Helpfulness'}, {'Quality'}, {'Spatial-Awareness'}, {'Domain-Knowledge'} and the numbers are always floats between 0 and 1.

"""
ONE_MINUTE = 60
@sleep_and_retry
@limits(calls=100, period=ONE_MINUTE)
async def use_gpt_to_extract_json(row):
    # Initialize the OpenAI client

    # Extract the user prompt from the row
    if isinstance(row['gpt4v_response'], str):
        response_dict = json.loads(row['gpt4v_response'])
    else:
        response_dict = row['gpt4v_response']
    user_prompt = response_dict['choices'][0]['message']['content'] + 'Please extract the attibutes in the json format'

    # Create the payload for the request
    payload = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_message_json_extraction_version_2},
            {"role": "user", "content": user_prompt}
        ]
    }

    # Send the request and get the response
    response = await client.chat.completions.create(**payload)

    # Return the JSON response
    return response.choices[0].message.content

# allow for 100 concurrent entries within a 60 second window
# rate_limit = AsyncLimiter(100, 60)
async def rate_limited_use_gpt_to_extract_json(text_dict, index, progress_tracker):
    result = await use_gpt_to_extract_json(text_dict)
    progress_tracker['completed'] += 1
    print(f"Progress: {progress_tracker['completed']}/{progress_tracker['total']} tasks completed.")
    return index, result
    
async def rate_limited_as_completed(merged_data):
    completed = {}
    progress_tracker = {'completed': 0, 'total': len(merged_data)}

    tasks_get_responses = [rate_limited_use_gpt_to_extract_json(value_dict, key, progress_tracker) for key, value_dict in merged_data.items()]
    for future in asyncio.as_completed(tasks_get_responses):
        index, response = await future
        completed[index] = response
    return completed
# Time the following code
start_time = time.time()
output_dict = asyncio.run(rate_limited_as_completed(merged_data))
end_time = time.time()
print(f'Total time taken: {end_time - start_time} seconds')
print(f'How many rows are extracted: {len(output_dict)} out of {len(merged_data)}')
# Write the final output to the file
with open(output_path, 'w') as f:
    json.dump(output_dict, f)
