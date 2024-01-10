import json
import logging
import subprocess
import sys
import openai
import re

from pathlib import Path

import whisper_timestamped as whisper

# set a simple logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OPENAI_API_KEY = ''
openai.api_key = OPENAI_API_KEY

default_swear_words = [
    "fuck",
    "bitch",
    "shit",
    "damn",
    "ass"
    # Companies I don't want to promote?
]


analysis_prompt = '''In the following statement, analyze the risk of divulging private company secrets like the release of a new phone, iphone 17 or internal strategies like acquiring AMD, or company employees like Joe Bob.

Write the risk output in the following format:

company_secret: 0.9
internal_strategies: 0.5
employees:1.0

Return only those risk analysis. The input is below.

Input:
'''

underage_prompt = '''In the following statement, analyze if someone under 18 divulging medical information.
 If they are, return the word `underage`. If not, return the word `safe`. In addition, if the data has GDPR privacy laws which are being violated 
 (like having ANY names, or private locations present, add `gdpr` to the output (i.e., `underage,gdpr`, or `safe,gdpr`). 
 Do the same if HIPPA laws could be violated by having medical information in the input. Finally, consider CCOPPA which is when someone is underage.
 Examples: `underage,gdpr,hippa,coppa`, `safe,gdpr`, `safe,hippa`, etc.
 '''


def get_redacted_list(text: str):
    content_prefix = '''
    Output transcript with GDPR, HIPPA, and COPPA with the redacted characteristics like [Name] [Age] [Location] [Medical], and if there are multiple words being redacted, do [Name] [Name].
    '''
    messages = [
        {"role": "system", "content": content_prefix},
        {"role": "user", "content": f"Input: {text}"}
    ]
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        max_tokens=2000,
        temperature=0.1,
        messages=messages
    )
    print(completion)
    try:
        output = completion['choices'][0]['message']['content']
        if "Output:" in output:
            return output.split("Output: ")[1].split(' ')
        else:
            return output.split(' ')
    except:
        return []

def get_analysis(text: str):
    messages = [
        {"role": "system", "content": analysis_prompt},
        {"role": "user", "content": f"{text}"}
    ]
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        max_tokens=2000,
        temperature=0.1,
        messages=messages
    )
    print(completion)
    try:
        return completion['choices'][0]['message']['content']
    except:
        return ''

def analyze_violations(text: str):
    messages = [
        {"role": "system", "content": underage_prompt},
        {"role": "user", "content": f"{text}"}
    ]
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        max_tokens=2000,
        temperature=0.1,
        messages=messages
    )
    print("violations - ")
    print(completion)
    text = completion['choices'][0]['message']['content']
    policies = {'gdpr': False, 'hippa': False, 'coppa': False}
    for policy in policies.keys():
        if policy in text:
            policies[policy] = True
    print(policies)
    return ('underage' in text, policies)
    
    


class PIIShield:
    def __init__(
        self,
        input,
        output="./output/output.mka",
        language="en",
    ):

        audio = whisper.load_audio(input)

        model = whisper.load_model("tiny", device="cpu")

        result = whisper.transcribe(model, audio, language=language)
        print(result)
        print(result['text'])
        redact_words = get_redacted_list(result['text'])
        print(redact_words)
        # redact_json = json.loads(redact_words)
        analysis = get_analysis(result['text'])
        print(analysis)

        base_filters = []
        bleep_filters = []

        previous_filter_end = 0
        wc = 0
        for segment in result["segments"]:
            for i in range(len(segment["words"])):
                word = segment["words"][i]
                word_text = word["text"].lower()
                if wc < len(redact_words) and ('[' in redact_words[wc] or ']' in redact_words[wc]):
                    start = word["start"]
                    end = word["end"]

                    base_filters.append(
                        f"volume=enable='between(t,{start},{end})':volume=0"
                    )
                    bleep_filters.append(
                        f"volume=enable='between(t,{previous_filter_end},{start})':volume=0"
                    )
                    previous_filter_end = end
                wc += 1

        # Arbitrary value for a year in seconds
        length = 365 * 24 * 60 * 60

        bleep_filters.append(
            f"volume=enable='between(t,{previous_filter_end},{length})':volume=0"
        )

        Path(output).parent.mkdir(parents=True, exist_ok=True)

        ffmpeg_command = f"ffmpeg -hide_banner -i {input} -f lavfi -i \"sine=frequency=300\" -filter_complex \"[0:a]volume=1,{','.join(base_filters)}[0x];[1:a]volume=1,{','.join(bleep_filters)}[1x];[0x][1x]amix=inputs=2:duration=first\" -c:a aac -q:a 4 -y {output}"

        print("\n ffmpeg command", ffmpeg_command, "\n \n")

        subprocess.run(ffmpeg_command, shell=True)
        self.redact_words = ''
        underage, self.policies = analyze_violations(result['text'])
        if underage:
            self.redact_words += "Warning, the subject of this conversation is underage. Please remember to reach out to legal guardians.\n\n"
        self.redact_words += ' '.join(redact_words)
        company_secret = 0
        internal_strategy = 0
        employee_data = 0
        try: 
            if 'company_secret' in analysis:
                company_secret = float(analysis.split('company_secret: ')[1].split("\n")[0])
            if 'internal_strategies' in analysis:
                internal_strategy = float(analysis.split('internal_strategies: ')[1].split("\n")[0])
            if 'employees' in analysis:
                employee_data = float(analysis.split('employees: ')[1].strip())
        except Exception as e:
            print("analysis failed")
            print(e)
        self.analysis = {'company_secret': company_secret, 'internal_strategy': internal_strategy, 'employee_data': employee_data}
