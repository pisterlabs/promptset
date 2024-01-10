from pathlib import Path
import openai
from pdf_extractor import PDFDatesFinder
from pdf_extractor import PDFDeductiblesFinder
from postprocessing_functions import postprocess_and_clean_dates
from tqdm import tqdm
import json
from datetime import datetime
import time

# OpenAI model settings
API_KEY = Path("credentials/openai_api_key.txt").read_text().strip()
openai.api_key = API_KEY
model_id4 = 'gpt-4-1106-preview'


# time start
start = datetime.now()


# ChatGPT-4
def ChatGPT4_conversation(conversation):
    response = openai.ChatCompletion.create(
        model=model_id4,
        messages=conversation
    )
    conversation.append({'role': response.choices[0].message.role, 
                         'content': response.choices[0].message.content})
    return conversation




def compose_prompt_date(
    text: str,
) -> str:
    prompt = f"""\
You are tasked with finding start date and end date in the text.

Notice that the difference between the start date and end date USUALLY is a multiple of 6 months.

If dates ARE found, output in the following format:
```
Start date: YYYY-MM-DD
End date: YYYY-MM-DD
```
If dates ARE NOT found, output the following:
```
Start date: N/A
End date: N/A
```

Output start date and end date from the following text:

```
{text}
```\
"""

    return prompt


def compose_prompt_deductibles(
    text: str,
) -> str:
    prompt = f"""\
You are tasked with finding all the mentioned deductibles in the text.

Notice that the text MAY NOT contain deductibles.

Your output MUST be a list of deductibles, each in the following format:
```
<deductible name>: <deductible value>
```

If NO deductibles are mentioned, output the following:
```
Not found
```

Output deductibles from the following text:

```
{text}
```\
"""

    return prompt


# Insurances folder path
insurances_folder_path = Path('insurances')
output_dictionary = {}

# Iterate over all files in the insurances folder
for file_path in insurances_folder_path.iterdir():
    
    try:
        # file name and current time
        print(f"File name: {file_path.stem} | Time: {datetime.now()}")

        if file_path.is_file():
            
            # Identification dates
            extractor_dates = PDFDatesFinder(file_path, API_KEY)
            pages = extractor_dates.extract_text()
            text_pages_dates, index_pages_dates = extractor_dates.identify_period_pages(pages)

            # Dates extraction using GPT
            output_dates = []

            for page in text_pages_dates:
                    
                    conversation = []

                    prompt = compose_prompt_date(
                        text=page,
                    )
                    conversation.append({'role': 'user', 'content': prompt})
                    conversation = ChatGPT4_conversation(conversation)

                    response = conversation[-1]['content'].strip()

                    if response:
                        output_dates.append(response)

            # Postprocessing dates
            output_dates = postprocess_and_clean_dates(output_dates)
            
            
            # Identification deductibles
            extractor_deductibles = PDFDeductiblesFinder(file_path, API_KEY)
            pages = extractor_deductibles.extract_text()
            text_pages_deductibles, index_pages_deductibles = extractor_deductibles.identify_deductibles_pages(pages)

            # Deductibles extraction using GPT
            output_deductibles = []

            for page in text_pages_deductibles:
                    
                    conversation = []

                    prompt = compose_prompt_deductibles(
                        text=page,
                    )
                    conversation.append({'role': 'user', 'content': prompt})
                    conversation = ChatGPT4_conversation(conversation)

                    response = conversation[-1]['content'].strip()

                    if response:
                        output_deductibles.append(response)

        output_dictionary[file_path.stem] = {'insurance period': output_dates, 'deductibles': output_deductibles}
    
    except Exception as e:
        print(f"Error with file: {file_path.stem}")
        print(e)

    time.sleep(120)

# time end
end = datetime.now()

# Save output dictionary as json file
with open('output.json', 'w') as fp:
    json.dump(output_dictionary, fp)

    print(f"Time taken: {end-start}")

        


