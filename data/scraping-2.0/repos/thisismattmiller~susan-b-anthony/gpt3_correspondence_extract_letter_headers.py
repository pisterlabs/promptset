import os
import openai
import glob
import json
import util

# openai.api_key = os.getenv("OPENAI_API_KEY")



for file in glob.glob('anthony-correspondence/*.json'):
  print(f'doing {file}')
  data = json.load(open(file))

  if 'full_text' in data:
    full_text = util.clean_up_transcribed_text(data['full_text'])

    if 'gpt' not in data:
      data['gpt'] = {}

    if 'correspondence-headers' not in data['gpt']:


      response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"From what geographical place and what date and to who and what was the salutation was this letter written? Return your answer in JSON using the keys \"geographical_place\",  \"date\", \"recipient\", \"salutation\":\n\n---\n{full_text}\n---\n",    
        temperature=0.7,
        max_tokens=250,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
      )

      data['gpt']['correspondence-headers'] = response['choices'][0]

      json.dump(data,open(file,'w'),indent=2)

    