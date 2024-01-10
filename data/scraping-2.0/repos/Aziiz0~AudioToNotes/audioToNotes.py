import os
import mutagen
import logging
import glob
import openai
import textwrap
import re
import json
import wave
import subprocess
import whisper
import torch
from transformers import GPT2Tokenizer
from transformers import pipeline
from notion_client import Client
from datetime import datetime
import nltk
from math import ceil

def convert_and_process_mp4_to_wav(mp4_file_path, speed_factor=1.25):
    # Ensure the wav directory exists
    if not os.path.isdir("wav"):
        os.mkdir("wav")

    # Construct the wav file path
    filename = os.path.basename(mp4_file_path)
    wav_file_path = f"wav/{filename[:-4]}.wav"
    
    # Returning function if wav file already exists
    if os.path.exists(wav_file_path):
        return wav_file_path
    
    # Begin the conversion process
    print("Converting MP4 to WAV and applying filters...")
    command = [
        'ffmpeg', 
        '-i', mp4_file_path, 
        '-vn', 
        '-acodec', 'pcm_s16le',
        '-ac', '1',
        '-ar', '16k',
        '-af', f'silenceremove=1:0:-40dB,atempo={speed_factor},volume=1.5',
        wav_file_path
    ]

    process = subprocess.Popen(command, stderr=subprocess.PIPE, universal_newlines=True)

    duration = None
    for line in process.stderr:
        # get duration for progress calculation
        if duration is None:
            match = re.search(r"Duration: (\d+):(\d+):(\d+).(\d+)", line)
            if match is not None:
                hours = int(match.group(1))
                minutes = int(match.group(2))
                seconds = int(match.group(3))
                duration = hours * 3600 + minutes * 60 + seconds
                print(f"File duration: {duration} seconds")

        # get progress
        match = re.search(r"time=(\d+):(\d+):(\d+).(\d+)", line)
        if match is not None and duration is not None:
            hours = int(match.group(1))
            minutes = int(match.group(2))
            seconds = int(match.group(3))
            progress = hours * 3600 + minutes * 60 + seconds
            print(f"Progress: {progress / duration * 100:.2f}%")

    return wav_file_path

def get_duration(file):
    try:
        # Get the tmp file path
        filePath = file

        if not filePath:
            raise Exception("File path is missing or invalid.")

        # Parse the file with mutagen
        try:
            dataPack = mutagen.File(filePath, easy=True)
        except Exception as e:
            raise Exception("Failed to read audio file metadata. The file format might be unsupported or corrupted, or the file might no longer exist at the specified file path (which is in temp storage).") from e

        # Get and return the duration in seconds
        duration = round(dataPack.info.length)
        return duration
    except Exception as error:
        # Log the error and return an error message or handle the error as required
        logging.error(error)
        raise Exception(f"An error occurred while processing the audio file: {str(error)}")

def transcribe_audio(file_path):
    def check_if_transcript_exists(file_path):
        # Remove the extension from the file path
        base_name = os.path.splitext(file_path)[0]

        # Append -transcript.txt to the file path
        transcript_file_path = base_name + "-transcript.txt"

        return os.path.isfile(transcript_file_path), transcript_file_path

    # Gather if transcription already exists and its path
    isMade, transcript_file_path = check_if_transcript_exists(file_path)

    # Check if path exists
    if isMade:
        # Open path and return contents
        with open(transcript_file_path) as trans_file:
            return trans_file.read()

    # Since it doesn't exist, create transcript
    model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")
    result = model.transcribe(file_path)

    # Write transcript to file
    with open(transcript_file_path, "w") as transcript_file:
        transcript_file.write(result["text"])

    return result["text"]

def openai_chat(transcript, file_path, model):
    def calculate_token_size(text):
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokens = tokenizer.encode(text)
        return len(tokens)
    
    def format_chat(result, total_tokens):
        def clean_gpt_output(gpt_output):
            def remove_trailing_commas(json_string):
                regex = r",\s*(?=])"
                return re.sub(regex, '', json_string)
            
            def parse_json(json_string):
                try:
                    return json.loads(json_string)
                except Exception as e:
                    print(f"Error while parsing cleaned JSON string:\n {e}")
                    print(f"Cleaned JSON string:\n {json_string}")
                    return {}
            
            return parse_json(remove_trailing_commas(gpt_output))

        def array_sum(arr):
            return sum(arr)

        def remove_null(d):
            return {k: (remove_null(v) if isinstance(v, dict) else v) for k, v in d.items() if v is not None}
        
        results_array = []
        usageNum = 0
        for results in result:
            json_string = results \
                .replace("^[^\{]*?{", "{") \
                .replace("\}[^}]*?$", "}") \
                .replace("\\", "\\\\") \
                .replace("\\\\\\\\", "\\\\")

            json_obj = clean_gpt_output(json_string)
            
            response = {
                'choice': json_obj,
                'usage': total_tokens
            }
            usageNum += 1
            results_array.append(response)
            
        chat_response = {
            'title': [],
            'summary': [],
            'main_points': [],
            'action_items': [],
            'example_problems': [],
            'follow_up': [],
            'related_topics': [],
            'usageArray': []
        }
        
        for arr in results_array:
            chat_response['title'].append(arr['choice']['title'])
            chat_response['summary'].append(arr['choice']['summary'])
            chat_response['main_points'].extend(arr['choice']['main_points'])
            chat_response['action_items'].extend(arr['choice']['action_items'])
            chat_response['example_problems'].extend(arr['choice']['example_problems'])
            chat_response['follow_up'].extend(arr['choice']['follow_up'])
            chat_response['related_topics'].extend(arr['choice']['related_topics'])
            chat_response['usageArray'].append(arr['usage'])
        
        final_chat_response = {
            'title': chat_response['title'],
            'summary': ' '.join(chat_response['summary']),
            'main_points': chat_response['main_points'],
            'action_items': chat_response['action_items'],
            'example_problems': chat_response['example_problems'],
            'follow_up': chat_response['follow_up'],
            'related_topics': chat_response['related_topics'],
            'tokens': array_sum(chat_response['usageArray'])
        }
        #final_chat_response = remove_null(final_chat_response)

        return final_chat_response
    
    def check_if_summary_exists(file_path):
        # Remove the extension from the file path
        base_name = os.path.splitext(file_path)[0]
        
        # Append -summary.txt to the file path
        summary_file_path = base_name + "-summary.json"

        return os.path.isfile(summary_file_path), summary_file_path
    
    # Gather if summary already exists and its path
    isMade, summary_file_path = check_if_summary_exists(file_path)
    
    # Check if path exists
    if isMade:
        # Open path and return contents
        with open(summary_file_path, 'r') as summary_file:
            return json.load(summary_file)
    
    # max tokens
    max_tokens_3_5 = 4000
    max_tokens_3_5_16k = 16000
    max_tokens_4 = 8000
    max_tokens_4_32k = 32000
    max_responses = 4
    
    system_message = """
    You are an assistant that only speaks JSON and TeX equations. Do not write normal text.

    Example formatting:

    {
        "title": "Math is Fun",
        "summary": "Why we love math, math is great!",
        "action_items": [
            "Study math",
            "Learn what the set $$\\Z$$ means",
            "learn the integral $$\\int^{1}_{0} x \\,dx$$ equals"
        ],
        "follow_up": [
            "item 1",
            "item 2",
            "item 3"
        ],
        "example_problems": [
            "Given that $$c\\in\\Z_{6}$$ what are the elements of $$c$$?",
            "item 2",
            "item 3"
        ],
        "related_topics": [
            "item 1",
            "item 2",
            "item 3"
        ]
    }
    """
    user_message = '''
    Analyze the transcript provided below, then provide the following:
        Key "title:" - add a title.
        Key "summary" - create a summary.
        Key "main_points" - add an array of the main points using TeX equations. Limit each item to 100 words, and limit the list to 6 items.
        Key "action_items:" - add an array of action items using TeX equations. Limit each item to 50 words, and limit the list to 3 items.
        Key "follow_up:" - add an array of follow-up questions using TeX equations. Limit each item to 50 words, and limit the list to 3 items.
        Key "example_problems:" - add an array of example problems from the transcript using TeX equations. Limit each item to 200 words, and limit the list to 3 items.
        Key "related_topics:" - add an array of topics related to the transcript. Limit each item to 50 words, and limit the list to 3 items.

        Ensure that the final element of any array within the JSON object is not followed by a comma.

        Transcript:
    '''

    def even_chunk_text(text, max_tokens, max_responses):
        words = text.split(' ')  # split text into words
        total_words = len(words)

        if total_words <= max_tokens:  # if total words is less than or equal to max tokens, no need to chunk
            return [text]

        chunk_count = max_responses  # initially try to divide the text into max_responses chunks
        while True:
            words_per_chunk = ceil(total_words / chunk_count)  # calculate the number of words per chunk

            # test if any chunk will exceed max_tokens
            if words_per_chunk > max_tokens:
                chunk_count += 1  # if so, increase the chunk count and try again
            else:
                break  # if not, we've found a good chunk count

        chunks = []
        for i in range(0, total_words, words_per_chunk):
            chunk = ' '.join(words[i:i+words_per_chunk])
            chunks.append(chunk)

        return chunks

    def split_transcript(system_message, user_message, transcript, max_tokens):
        # Calculate tokens used by the system and user message
        system_message_tokens = calculate_token_size(system_message)
        user_message_tokens = calculate_token_size(user_message)
        
        # Calculate remaining tokens for the transcript
        remaining_tokens = max_tokens - system_message_tokens - user_message_tokens
        
        # Split transcript into shorter strings if needed, based on remaining tokens
        strings_array = even_chunk_text(transcript, remaining_tokens, max_responses)
        
        '''itter = 1
        for thing in strings_array:
            print(f"{itter}. {thing}")
            itter += 1'''

        return strings_array

    def send_to_chat(strings_array, models):
        total_tokens = 0
        results_array = []
        index = 1
        for arr in strings_array:
            # Define the prompt
            prompt =f"""
            {user_message}
            {arr}
            """
            retries = 3
            print(f"Sending transcript part #{index}...")
            while retries > 0:
                try:
                    completion = openai.ChatCompletion.create(
                        model=models,
                        messages= [ {"role": "system", "content": system_message}, {"role": "user", "content": prompt} ],
                        temperature=0.2,
                    )
                    if completion['choices'][0]['finish_reason'] == "stop":
                        print(f"Sucessfully gathered summary #{index}!")
                    else:
                        print(f"Summary #{index} resulted in GPT length finish!\nLowering MAX_TOKEN by 500.")
                        return results_array, 0, total_tokens
                    
                    # Update result array and total tokens
                    results_array.append(completion['choices'][0]['message']['content'])
                    total_tokens += completion['usage']['total_tokens']
                    
                    break
                except openai.error.OpenAIError as e:
                    retries -= 1
                    if retries == 0:
                        print("Failed to get a response from OpenAI Chat API after 3 attempts.\n Lowering MAX_TOKEN by 500.")
                        return results_array, 0, total_tokens
                    print(f"OpenAI Chat API returned an error: {str(e)}. Retrying...")
            index += 1
        return results_array, 1, total_tokens
    
    gotResponses = 0
    while gotResponses == 0:
        if model == 'gpt-3.5-turbo':
            strings_array = split_transcript(system_message, user_message, transcript, max_tokens_3_5)
            result, gotResponses, total_tokens = send_to_chat(strings_array, model)
            if gotResponses == 0:
                max_tokens_3_5 = max_tokens_3_5 - 500
        elif model == 'gpt-3.5-turbo-16k':
            strings_array = split_transcript(system_message, user_message, transcript, max_tokens_3_5_16k)
            result, gotResponses, total_tokens = send_to_chat(strings_array, model)
            if gotResponses == 0:
                max_tokens_3_5_16k = max_tokens_3_5_16k - 500
        elif model == 'gpt-4':
            strings_array = split_transcript(system_message, user_message, transcript, max_tokens_4)
            result, gotResponses, total_tokens = send_to_chat(strings_array, model)
            if gotResponses == 0:
                max_tokens_4 = max_tokens_4 - 500
        elif model == 'gpt-4-32k':
            strings_array = split_transcript(system_message, user_message, transcript, max_tokens_4_32k)
            result, gotResponses, total_tokens = send_to_chat(strings_array, model)
            if gotResponses == 0:
                max_tokens_4_32k = max_tokens_4_32k - 500
        else:
            raise ValueError(f'Unsupported model: {model}')
    json_array = format_chat(result, total_tokens)

    with open(summary_file_path, 'w') as summary_file:
        json.dump(json_array, summary_file)

    return json_array

def local_summarize_text(transcript, file_path):
    # Initialize the summarization, question-answering, and text-generation pipelines
    summarizer = pipeline("summarization")
    question_answering = pipeline("question-answering")
    text_generator = pipeline("text-generation")

    def extract_information(text):
        # Generate a summary
        summary = summarizer(text, max_length=100, min_length=30)

        # Extract key points by asking questions
        key_points = question_answering({
            'question': 'What are the key points?',
            'context': text
        })

        # Extract example problems by asking questions
        example_problems = question_answering({
            'question': 'What are the example problems?',
            'context': text
        })

        # Extract action lists by generating text
        action_list = text_generator("The actions we need to take are", max_length=100)

        return summary, key_points, example_problems, action_list

    text = transcript
    summary, key_points, example_problems, action_list = extract_information(text)
    
    # Remove the extension from the file path
    base_name = os.path.splitext(file_path)[0]

    # Append -transcript.txt to the file path
    summary_file_path = base_name + "-summary.txt"
    
    # Write summary to file
    with open(summary_file_path, "w") as summary_file:
        summary_file.write("Summary: "+summary+"\nKey Points: "+key_points+"\nExample Problems: "+example_problems+"Action List: "+action_list)
    
    print(f"Summary: {summary}")
    print(f"Key Points: {key_points}")
    print(f"Example Problems: {example_problems}")
    print(f"Action List: {action_list}")

def make_paragraphs(transcript, summary):
    # Convert our json output into string, and ensure that there are not weird quotes
    summary_string = json.dumps(summary)
    if len(summary_string) >= 2 and summary_string[0] == '"' and summary_string[-1] == '"':
        summary_string = summary_string[1:-1]
    
    
    # Assume `transcription` and `summary` are strings containing the transcript and summary respectively
    def sentence_grouper(arr, sentences_per_paragraph):
        return [' '.join(arr[i:i+sentences_per_paragraph]) for i in range(0, len(arr), sentences_per_paragraph)]

    def char_max_checker(arr):
        return [item for sublist in [textwrap.wrap(element, 800) if len(element) > 800 else [element] for element in arr] for item in sublist]

    # Tokenize sentences
    transcript_sentences = nltk.sent_tokenize(transcript)
    summary_sentences = nltk.sent_tokenize(json.dumps(summary).replace('"', ''))

    sentences_per_paragraph = 3

    # Group sentences into paragraphs
    paragraphs = sentence_grouper(transcript_sentences, sentences_per_paragraph)
    summary_paragraphs = sentence_grouper(summary_sentences, sentences_per_paragraph)

    # Check and adjust paragraph lengths
    length_checked_paragraphs = char_max_checker(paragraphs)
    length_checked_summary_paragraphs = char_max_checker(summary_paragraphs)

    all_paragraphs = {
        'transcript': length_checked_paragraphs,
        'summary': length_checked_summary_paragraphs
    }

    return all_paragraphs

def to_notion(result, all_paragraphs, file, model):
    # Initialize a new Notion client
    notion = Client(auth=notion_token)

    # Set the Database ID
    dbID = db_id
    
    # Your title to check
    split_path = os.path.splitext(file)[0].split("\\")[-1]
    if 'L' in split_path:
        file_name = split_path.split('L')[1]
    else:
        print(f"Unexpected filename format: {split_path}")
        file_name = input("Please enter a numer for the file: ")
    result['name'] = file_name
    
    # Current OpenAPI pricing (whisper is per-minute, gpt is per 1k tokens)
    if model == 'gpt-3.5-turbo':
        gptRateIn = 0.0015
        gptRateOut = 0.002
    elif model == 'gpt-3.5-turbo-16k':
        gptRateIn = 0.003
        gptRateOut = 0.004
    elif model == 'gpt-4':
        gptRateIn = 0.03
        gptRateOut = 0.06
    elif model == 'gpt-4-32k':
        gptRateIn = 0.06
        gptRateOut = 0.12
    else:
        gptRateIn = 0
        gptRateOut = 0
    gptRate = (gptRateIn + gptRateOut)/2
    whisperRate = 0
    

    # Get the file duration
    duration = get_duration(file)

    # Get the Date
    today = datetime.now()
    year = today.year
    month = str(today.month).zfill(2)
    day = str(today.day).zfill(2)

    date_str = f"{year}-{month}-{day}"

    # Build an object with all the content from the Chat API response
    meta = result

    # Add the array of transcript paragraphs
    meta['transcript'] = all_paragraphs['transcript']

    # Add the paragraph-separated summary
    meta['long_summary'] = all_paragraphs['summary']

    # Add cost values
    transcriptionCost = (duration / 60) * whisperRate
    meta['transcription-cost'] = f"Transcription Cost: ${transcriptionCost:.3f}"
    chatCost = (meta['tokens'] / 1000) * gptRate
    meta['chat-cost'] = f"Chat API Cost: ${chatCost:.3f}"
    totalCost = transcriptionCost + chatCost
    meta['total-cost'] = f"Total Cost: ${totalCost:.3f}"

    # Start building the data object that will be sent to Notion
    data = {
    "parent": {
        "type": "database_id",
        "database_id": dbID
      },
      "icon": {
        "type": "emoji",
        "emoji": "🤖"
      },
      "properties": {
        "Title": {
          "title": [
            {
              "text": {
                "content": meta['name']+". "+" | ".join(meta['title'])
              }
            }
          ]
        },
        "Type": {
          "select": {
            "name": "AI Transcription"
          }
        },
        "AI Cost": {
          "number": round(totalCost * 1000) / 1000
        },
        "Duration (Seconds)": {
          "number": duration
        }
      },
      "children": [
        {
          "callout": {
            "rich_text": [
              {
                "text": {
                  "content": "This AI transcription and summary was created on "
                }
              },
              {
                "mention": {
                  "type": "date",
                  "date": {
                    "start": date_str
                  }
                }
              },
              {
                "text": {
                  "content": ". "
                }
              }
            ],
            "icon": {
              "emoji": "🤖"
            },
            "color": "blue_background"
          }
        },
        {
          "table_of_contents": {
            "color": "default"
          }
        },
        {
          "heading_1": {
            "rich_text": [
              {
                "text": {
                  "content": "Summary"
                }
              }
            ]
          }
        }
      ]
    }
    
    # Construct the summary
    for paragraph in meta['long_summary']:
        summary_paragraph = {
            "paragraph": {
              "rich_text": [
                {
                  "text": {
                    "content": paragraph
                  }
                }
              ]
            }
        }
        data['children'].append(summary_paragraph)
    
    # Add the Transcript header
    transcript_header = {
      "heading_1": {
        "rich_text": [
          {
            "text": {
              "content": "Transcript"
            }
          }
        ]
      }
    }
    data['children'].append(transcript_header)
    
    # Create an array of paragraphs from the transcript
    # If the transcript has more than 80 paragraphs, I need to split it and only send
    # the first 80.
    transcriptHolder = []
    transcriptBlockMaxLength = 80
    
    for i in range(0, len(meta['transcript']), transcriptBlockMaxLength):
        chunk = meta['transcript'][i:i + transcriptBlockMaxLength]
        transcriptHolder.append(chunk)

    # Push the first block of transcript chunks into the data object
    firstTranscriptBlock = transcriptHolder[0]
    #print(firstTranscriptBlock)
    for sentence in firstTranscriptBlock:
        paragraphBlock = {
            "paragraph": {
                "rich_text": [
                    {
                        "text": {
                            "content": sentence
                        }
                    }
                ]
            }
        }
        #print(sentence)
        data['children'].append(paragraphBlock)

    # Add Additional Info
    additionalInfoArray = []

    additionalInfoHeader = {
        "heading_1": {
            "rich_text": [
                {
                    "text": {
                        "content": "Additional Info"
                    }
                }
            ]
        }
    }

    additionalInfoArray.append(additionalInfoHeader)

    # Add Action Items
    def additionalInfoHandler(arr, header, itemType):
        infoHeader = {
            "heading_2": {
                "rich_text": [
                    {
                        "text": {
                            "content": header
                        }
                    }
                ]
            }
        }

        additionalInfoArray.append(infoHeader)

        if header == "Arguments and Areas for Improvement":
            argWarning = {
                "callout": {
                    "rich_text": [
                        {
                            "text": {
                                "content": "These are potential arguments and rebuttals that other people may bring up in response to the transcript. Like every other part of this summary document, factual accuracy is not guaranteed."
                            }
                        }
                    ],
                    "icon": {
                        "emoji": "⚠️"
                    },
                    "color": "orange_background"
                }
            }

        if isinstance(arr, list):
            for item in arr:
                infoItem = {
                    itemType: {
                        "rich_text": [
                            {
                                "text": {
                                    "content": item
                                }
                            }
                        ]
                    }
                }

                additionalInfoArray.append(infoItem)
        else:
            print(f"additionalInfoHandler: arr is not an array: {arr}")

    additionalInfoHandler(meta['main_points'], "Main Points", "bulleted_list_item")
    additionalInfoHandler(meta['example_problems'], "Example Problems", "bulleted_list_item")
    additionalInfoHandler(meta['action_items'], "Potential Action Items", "to_do")
    additionalInfoHandler(meta['follow_up'], "Follow-Up Questions", "bulleted_list_item")

    additionalInfoHandler(meta['related_topics'], "Related Topics", "bulleted_list_item")
    
    # Add sentiment and cost
    metaArray = [meta['transcription-cost'], meta['chat-cost'], meta['total-cost']]
    additionalInfoHandler(metaArray, "Meta", "bulleted_list_item")

    #input(f"About to send dir:\n{data}")

    # Create the page in Notion
    page = notion.pages.create(**data)

    # Create an object to pass to the next step
    responseHolder = {
        "response": page,
        "transcript": transcriptHolder,
        "additional_info": additionalInfoArray
    }

    return responseHolder

def update_notion(responseHolder):
    def send_transcript_to_notion(notion, page_id, transcript):
        data = {
            'block_id': page_id,
            'children': []
        }

        for sentence in transcript:
            paragraph_block = {
                'paragraph': {
                    'rich_text': [
                        {
                            'text': {
                                'content': sentence
                            }
                        }
                    ]
                }
            }
            data['children'].append(paragraph_block)

        response = notion.blocks.children.append(**data)
        return response

    def send_additional_info_to_notion(notion, page_id, info):
        data = {
            'block_id': page_id,
            'children': []
        }

        for block in info:
            data['children'].append(block)

        response = notion.blocks.children.append(**data)
        return response

    def run(steps):
        notion = Client(auth=notion_token)

        # Set the page ID
        page_id = steps['response']['id'].replace('-', '')

        # Send remaining Transcript blocks to the Notion Page
        transcript_array = steps['transcript']
        transcript_array.pop(0)
        transcript_addition_responses = []

        for transcript in transcript_array:
            response = send_transcript_to_notion(notion, page_id, transcript)
            transcript_addition_responses.append(response)

        # Send the Additional Info to the Notion Page
        additional_info = steps['additional_info']
        info_holder = []
        info_block_max_length = 95

        for i in range(0, len(additional_info), info_block_max_length):
            chunk = additional_info[i:i + info_block_max_length]
            info_holder.append(chunk)

        additional_info_addition_responses = []
        for addition in info_holder:
            response = send_additional_info_to_notion(notion, page_id, addition)
            additional_info_addition_responses.append(response)

        all_api_responses = {
            'transcript_responses': transcript_addition_responses,
            'additional_info_responses': additional_info_addition_responses
        }

        return all_api_responses

    allAPIResponses = run(responseHolder)
    
    return allAPIResponses

# Get OpenAi API key from Environment variable named OpenAi_API_Key
openai.api_key = os.environ.get('OpenAi_API_Key')

# Get Notion Token and Database ID from Environment variable named NOTION_TOKEN and NOTION_DATABASE_ID
notion_token = os.environ.get('NOTION_TOKEN')
db_id = os.environ.get('NOTION_DATABASE_ID')

def check_environment_variables():
    if not openai.api_key:
        print("OpenAI API Key is not set.")
        print("Please set it by adding the following line to your environment:")
        print("export OpenAi_API_Key='your_openai_api_key'")
    if not notion_token:
        print("Notion Token is not set.")
        print("Please set it by adding the following line to your environment:")
        print("export NOTION_TOKEN='your_notion_token'")
    if not db_id:
        print("Notion Database ID is not set.")
        print("Please set it by adding the following line to your environment:")
        print("export NOTION_DATABASE_ID='your_notion_database_id'")
    if not openai.api_key or not notion_token or not db_id:
        print("Please restart the program after setting the environment variables.")
        exit()

def choose_model():
    print("\nChoose a model:")
    print("1. gpt-3.5-turbo")
    print("2. gpt-3.5-turbo-16k")
    print("3. gpt-4")
    print("4. gpt-4-32k (NOT WORKING)")
    while True:
        choice = input("Enter your choice: ")
        if choice == '1':
            return 'gpt-3.5-turbo'
        elif choice == '2':
            return 'gpt-3.5-turbo-16k'
        elif choice == '3':
            return 'gpt-4'
        elif choice == '4':
            return 'gpt-4-32k'
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

def start_translation():
    # Choose the model
    model = choose_model()

    # Create array of all mp4 files
    mp4_files = glob.glob('*.mp4')

    # Convert mp4 to wav
    for file in mp4_files:
        wav_file_path = convert_and_process_mp4_to_wav(file)

    # Create array of all wav files
    wav_files = glob.glob('wav\*.wav')

    # Makes 'wav' directory if not already there
    if not os.path.exists('wav'):
        os.makedirs('wav')

    # Makes notesInNotion.txt if not already there
    if not os.path.exists('wav/notesInNotion.txt'):
        with open('wav/notesInNotion.txt', 'w') as f:
            pass

    for file in wav_files:
        file_name = os.path.splitext(file)[0].split("\\")[-1]
        if 'L' in file_name:
            text_name = file_name.split('L')[1]
        else:
            text_name = file_name  # or some other default value
        print(f"Beginning NOTE #{text_name}")
        
        # Get Transcript
        print("Getting transcription...")
        transcription = transcribe_audio(file)
        #input("Press Enter to summarize transcript...")

        # Get a summarization of Transcript
        print("Summarizing transcription...")
        result = openai_chat(transcription, file, model)  # pass the model here
        all_paragraphs = make_paragraphs(transcription, result['summary'])
        #result = local_summarize_text(transcription, file)
        #input("Press Enter to send to Notion...")
        
        with open('wav/notesInNotion.txt', 'a+') as notionFile:
            notionFile.seek(0)  # move the file pointer to the beginning of the file
            contents = notionFile.read()
            if text_name in contents:
                print(f"{text_name} is already in Notion.")
            else:
                print("Setting Notion up...")
                responseHolder = to_notion(result, all_paragraphs, file, model)
                #input("Press Enter to update Notion...")
                allAPIResponses = update_notion(responseHolder)
                notionFile.write(text_name + '\n')
        
        print(f"Finished with NOTE #{text_name}")
        #input("Press Enter to continue...")

def show_instructions():
    print("Instructions:")
    print("1. Place your .mp4 files in the same directory as this script.")
    print("2. The script will convert .mp4 files to .wav, generate transcripts and summaries, and upload them to Notion.")
    print("3. The summaries will be stored in a file 'notesInNotion.txt' in the 'wav' directory.")
    print("4. Make sure you have set your OpenAi_API_Key, NOTION_TOKEN, and NOTION_DATABASE_ID as environment variables.")

def menu():
    check_environment_variables()

    while True:
        print("\nMenu:")
        print("1. Start translation")
        print("2. Show instructions")
        print("3. Exit")
        choice = input("Enter your choice: ")
        if choice == '1':
            start_translation()
        elif choice == '2':
            show_instructions()
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

# Start the menu
menu()
