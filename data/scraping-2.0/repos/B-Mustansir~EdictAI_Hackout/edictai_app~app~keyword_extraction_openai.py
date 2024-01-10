import openai
from docx import Document
from .env import *

audio_file_path = "blob_images/Shorts.mp4"
openai.organization = generate_script_openai_organization
openai.api_key = generate_script_openai_api_key

def transcribe_audio(audio_file_path):
    with open(audio_file_path, 'rb') as audio_file:
        transcription = openai.Audio.transcribe("whisper-1", audio_file)
    return transcription['text']

def abstract_summary_extraction(transcription):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a highly skilled AI trained in language comprehension and summarization. I would like you to read the following text and summarize it into a concise abstract paragraph. Aim to retain the most important points, providing a coherent and readable summary that could help a person understand the main points of the discussion without needing to read the entire text. Please avoid unnecessary details or tangential points."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response['choices'][0]['message']['content']

def key_points_extraction(transcription):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a proficient AI with a specialty in distilling information into key points. Based on the following text, identify and list the main points that were discussed or brought up. These should be the most important ideas, findings, or topics that are crucial to the essence of the discussion. Your goal is to provide a list that someone could read to quickly understand what was talked about."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response['choices'][0]['message']['content']

def action_item_extraction(transcription):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are an AI expert in analyzing conversations and extracting action items. Please review the text and identify any tasks, assignments, or actions that were agreed upon or mentioned as needing to be done. These could be tasks assigned to specific individuals, or general actions that the group has decided to take. Please list these action items clearly and concisely."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response['choices'][0]['message']['content']

def sentiment_analysis(transcription):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "As an AI with expertise in language and emotion analysis, your task is to analyze the sentiment of the following text. Please consider the overall tone of the discussion, the emotion conveyed by the language used, and the context in which words and phrases are used. Indicate whether the sentiment is generally positive, negative, or neutral, and provide brief explanations for your analysis where possible."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response['choices'][0]['message']['content']

def meeting_minutes(transcription):
    abstract_summary = abstract_summary_extraction(transcription)
    key_points = key_points_extraction(transcription)
    action_items = action_item_extraction(transcription)
    sentiment = sentiment_analysis(transcription)
    return {
        'abstract_summary': abstract_summary,
        'key_points': key_points,
        'action_items': action_items,
        'sentiment': sentiment
    }

# transcription = transcribe_audio(audio_file_path)
# print(transcription)
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
print(meeting_minutes(news))