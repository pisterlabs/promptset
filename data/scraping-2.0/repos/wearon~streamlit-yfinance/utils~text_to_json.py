import openai
import streamlit as st


import json


def text_to_json(text):
    prompt = \
"""Extract the full text of a resume provided after json below into a JSON with exactly the following structure of JSON Resume:\n
{
  "basics": {
    "name": "",
    "label": "",
    "image": "",
    "email": "",
    "phone": "",
    "url": "",
    "summary": "",
    "location": {
      "address": "",
      "postalCode": "",
      "city": "",
      "countryCode": "",
      "region": ""
    },
    "profiles": [{
      "network": "Twitter",
      "username": "john",
      "url": "https://twitter.com/john"
    }]
  },
  "work": [{
    "name": "Company Name",
    "position": "Job Title",
    "url": "",
    "startDate": "",
    "endDate": "",
    "summary": "",
    "highlights": [
      "Started the company"
    ]
  }],

  "education": [{
    "institution": "",
    "url": "",
    "area": "",
    "studyType": "",
    "startDate": "",
    "endDate": "",
    "score": "4.0",
    "courses": [
      "DB1101 - Basic SQL"
    ]
  }],
  "awards": [{
    "title": "",
    "date": "",
    "awarder": "",
    "summary": ""
  }],
  "certificates": [{
    "name": "",
    "date": "",
    "issuer": "",
    "url": ""
  }],
#   "publications": [{
#     "name": "Publication",
#     "publisher": "Company",
#     "releaseDate": "2014-10-01",
#     "url": "https://publication.com",
#     "summary": "Description…"
#   }],
  "skills": [{
    "name": "Web Development",
    "level": "Master",
    "keywords": [
      "HTML",
      "CSS",
      "JavaScript"
    ]
  }],
  "languages": [{
    "language": "English",
    "fluency": "Native speaker"
  }],
  "interests": [{
    "name": "Wildlife",
    "keywords": [
      "Ferrets",
      "Unicorns"
    ]
  }],
  "references": [{
    "name": "",
    "reference": ""
  }],
  "projects": [{
    "name": "",
    "description": "",
    "highlights": [
      ""
    ],
    "keywords": [
      "HTML"
    ],
    "startDate": "2019-01-01",
    "endDate": "2021-01-01",
    "url": "https://project.com/",
    "roles": [
      "Team Lead"
    ],
    "entity": "Entity",
    "type": "application"
  }]
}

    
"""
    report = []
    res_box = st.empty()


    completions = openai.ChatCompletion.create( model="gpt-3.5-turbo", messages=[ {"role": "user", "content": prompt + '\n' + text}, ], temperature=1, n=1, stop=None, top_p=1, stream=True,)


    collected_events = []
    completion_text = ''
    something = ''
    for event in completions:
        collected_events.append(event)
        for choices in event['choices']:
            event_ntext = choices['delta']
            event_ntext = str(event_ntext)
            completion_text += event_ntext
            event_dict = json.loads(event_ntext)
            if "content" in event_dict:
                # st.write(event_dict["content"].replace("\n", "") ) 
                something += event_dict["content"].replace("", "")
                res_box.markdown(f'```\n{something}\n```')
            else :
                print(choices)
                st.success('Text to JSON conversion complete!')