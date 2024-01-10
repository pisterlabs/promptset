from urllib.parse import urlparse
from openai import OpenAI
import requests
import json
import os
import pprint
import google.generativeai as palm

with open('config.json', 'r') as c:
    config_data = json.load(c)
    generate_script_openai_organization = config_data['openai_organization'] 
    generate_script_openai_api_key = config_data['openai_api_key'] 

# palm.configure(api_key=f'{generate_script_api_key}')

# models = [m for m in palm.list_models(
# ) if 'generateText' in m.supported_generation_methods]
# model = models[0].name

# def generate_script(news):
#     # Prompt 1 for Creative Script Generation
#     prompt = f"""Imagine yourself as a charismatic news anchor, ready to captivate your audience with an engaging video script. Craft a script based on the following news: "{news}".

# Begin with a warm greeting and smoothly transition into highlighting the most significant and impactful points from the news article. Ensure that the script maintains an authentic and unbiased tone. Conclude the script by hinting at potential future developments, all within a video length of 60-90 seconds.

# Remember, your goal is to inform, inspire, and engage your viewers. Make it captivating and creative while staying true to the news story.

# Please break the script into meaningful chunks, each containing about 15-20 words, and separate them using <m>. """

#     # Generate the creative script
#     completion = palm.generate_text(
#         model=model,
#         prompt=prompt,
#         temperature=0,
#         max_output_tokens=2000,
#     )

#     # Prompt 2 for Chunk Extraction
#     prompt2 = f"""Take the creative script you generated earlier and remove any stars or keywords associated with them. Extract the 'chunk lines' from the following creative script:

# {completion.result}

# Ensure that each chunk consists of 15-20 words and conveys a distinct message or idea. Your output should be a series of these <m>-separated 'chunk lines' derived from the creative script."""

#     # Generate and return the extracted chunks
#     completion2 = palm.generate_text(
#         model=model,
#         prompt=prompt2,
#         temperature=0,
#         max_output_tokens=4000,
#     )
#     print('completion2/n')
#     print(completion2.result)
#     return completion2.result

# text = '''
# The Prime Minister, Shri Narendra Modi interacted with Team G20 at Bharat Mandapam today. The Prime Minister also addressed the gathering on the occasion.
# Speaking on the occasion, the Prime Minister underlined the accolades that are being showered for the successful organization of G20 and credited the ground level functionaries for this success.
# '''

# print(generate_script(text))


def generate_script(news):
    client = OpenAI(
        organization=generate_script_openai_organization,
        api_key=generate_script_openai_api_key,
    )

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "I want you to act as a Newsreader. I will provide you with a news article and you will create a script for to make a video out of it."},
            {"role": "user", "content": '''
        Ensure that the script maintains an authentic and unbiased tone. Consider the video length to be 60-90 seconds. Our goal is to inform viewers about the official news from the government, and engage the viewers to see news in a visual format. 
        Please break the script into meaningful chunks with independent meaning.
        Each chunk containing about 15-20 words.
        Separate these chunks using "<m>" in the output.  
        Note: Don't add any instructions or text in the output. Give the output in <m> tags only. 
        '''},
            {"role": "user", "content": f'''
        News article: {news}
        '''}
        ]
    )
    print(completion.choices[0].message.content)
    return (completion.choices[0].message.content)


news = '''
PM addresses Kaushal Dikshnat Samaroh 2023 via video message

“Kaushal Dikshnat Samaroh reflects the priorities of today's India”

“Country develops more with stronger youth power thereby doing justice to nation’s resources”

“Today, the whole world is of the belief that this century is going to be India's century”

“Our government understood the importance of skill and created a separate ministry for it, allocated separate budget”

“Important for industry, research and skill development institutions to be in tune with present times”

“Scope of skill development is continuously increasing in India. We are not limited to just mechanics, engineers, technology or any other service”

“Unemployment rate in India is at its lowest level in 6 years”

“IMF is confident of India becoming the top three economies of the world in the next 3-4 years”
Posted On: 12 OCT 2023 1:05PM by PIB Delhi
The Prime Minister, Shri Narendra Modi addressed the Kaushal Dikshnat Samaroh via video message today.

Addressing the gathering, the Prime Minister remarked that this festival of skill development is unique in itself and today’s event of joint convocation of skill development institutions across the country is a very commendable initiative. He said that Kaushal Dikshnat Samaroh reflects the priorities of today's India. Acknowledging the presence of thousands of youth connected with this event through technology, the Prime Minister conveyed his best wishes to all the youth.

Prime Minister Modi highlighted the importance of the power of the youth in utilizing the strengths of any country such as its natural or mineral resources, or its long coastlines and said that the country develops more with stronger youth power thereby doing justice to the nation’s resources. Today, the Prime Minister emphasized that a similar thinking is empowering India’s youth which is making unprecedented improvements in the entire ecosystem. “In this, the country's approach is two-pronged”, the Prime Minister said. He explained that India is preparing its youth to take advantage of new opportunities through skilling and education as he highlighted the new National Education Policy which has been established after almost 4 decades. The Prime Minister also underlined that the Government is establishing a large number of new medical colleges, and skill development institutes like IITs, IIMs or ITIs, and mentioned the crores of youth who have been trained under the Pradhan Mantri Kaushal Vikas Yojana. On the other hand, the Prime Minister stated that traditional sectors which provide jobs are also being strengthened while new sectors that promote employment and entrepreneurship are also being promoted. The Prime Minister also mentioned India making new records in goods exports, mobile exports, electronic exports, services exports, defence exports and manufacturing, and at the same time creating a large number of new opportunities for youth in many sectors such as space, startups, drones, animation, electric vehicles, semiconductors, etc.

“Today, the whole world is of the belief that this century is going to be India's century”, the Prime Minister said as he credited the young population of India for this. Shri Modi underlined that when the elderly population is increasing in many countries of the world, India is getting younger with each passing day. “India has this huge advantage”, he stressed as he noted the world looking towards India for its skilled youth. He informed that India's proposal regarding global skill mapping has been recently accepted at the G20 Summit which will help in creating better opportunities for youth in the coming times. The Prime Minister suggested not wasting any opportunity being created and assured that the Government is ready to support the cause. Shri Modi pointed out the neglect towards skill development in the previous governments and said, “Our government understood the importance of skill and created a separate ministry for it and allocated a separate budget.” He underlined that India is investing more in the skills of its youth than ever before and gave the example of Pradhan Mantri Kaushal Vikas Yojana which has strengthened the youth at the ground level. Under this scheme, the Prime Minister informed that about 1.5 crore youth have been trained so far. He further added that new skill centers are also being established near industrial clusters which will enable the industry to share its requirements with skill development institutes, thereby developing the necessary skill sets among the youth for better employment opportunities.

Highlighting the importance of skilling, upskilling and re-skilling, the Prime Minister noted the rapidly changing demands and nature of jobs and emphasized upgrading the skills accordingly. Therefore, the Prime Minister said, it is very important for industry, research and skill development institutions to be in tune with the present times. Noting the improved focus on skills, the Prime Minister informed that about 5 thousand new ITIs have been set up in the country in the last 9 years adding more than 4 lakh new ITI seats. He also mentioned that institutes are being upgraded as model ITIs with the objective of providing efficient and high-quality training along with best practices.

“The scope of skill development is continuously increasing in India. We are not limited to just mechanics, engineers, technology, or any other service”, the Prime Minister said as he mentioned that women’s self-help groups are being prepared for drone technology. Stressing the importance of Vishwakarmas in our everyday life, Shri Modi mentioned PM Vishwakarma Yojana which enables the Vishwakarmas to link their traditional skills with modern technology and tools.

The Prime Minister noted that new possibilities are being created for the youth as India's economy is expanding. He said that employment creation in India has reached a new height and the unemployment rate in India is at its lowest level in 6 years according to a recent survey. Noting that unemployment is decreasing rapidly in both rural and urban areas of India, the Prime Minister emphasized that the benefits of development are reaching both villages and cities equally, and as a result, new opportunities are increasing equally in both villages and cities. He also pointed out the unprecedented increase in the participation of women in India's workforce and credited the impact of the schemes and campaigns that have been launched in India in the past years regarding women empowerment.

Highlighting the recent figures released by the International Monetary Fund, the Prime Minister informed that India will remain the fastest-growing major economy in the coming years. He also recalled his resolve to take India among the top three economies of the world and said that IMF is also confident of India becoming the top three economies of the world in the next 3-4 years. He underlined that it would create new opportunities for employment and self-employment in the country. 

Concluding the address, the Prime Minister stressed making India the biggest center of skilled manpower in the world in order to provide smart and skilled manpower solutions. “The process of learning, teaching and moving forward should continue. May you be successful at every step in life”, the Prime Minister concluded. 
'''

# generate_script(news)
