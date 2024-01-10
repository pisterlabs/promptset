import os
import openai
openai.organization = ""
openai.api_key = ""
# print(openai.Model.list())

news = '''
NLC India Ltd secures 810 MW Grid Connected Solar Photovoltaic Power Project in Rajasthan
Posted On: 09 OCT 2023 12:34PM by PIB Delhi
NLC India Limited, a Navratna Central Public Sector Undertaking (CPSE) under the Ministry of Coal has won 810 MW Solar PV project capacity from Rajasthan Rajya Vidyut Nigam Limited (RRVUNL).

NLCIL has successfully garnered the entire capacity of the 810 MW tender floated by RRVUNL in December 2022, for developing the project RRVUNL’s 2000 MW Ultra Mega Solar Park at Pugal Tehsil, Bikaner District, Rajasthan. The Letter of Intent for this project has been issued by RRVUNL. This achievement marks a significant step forward in NLCIL's commitment to clean and sustainable energy solutions.

The land for the project and the power evacuation system connected to STU will be offered by RVUNL, paving the way for completion of the project at shorter period. This is the largest Renewable project to be developed by the company. With this project, the capacity of power project in   Rajasthan will be 1.36 GW including 1.1 GW of green power, bringing economies of scale and optimized fixed costs.

Considering the good Solar radiation at Rajasthan, the higher CUF for the project is possible and will generate green power of more than 50 Billion Units and offsets more than 50,000 tonnes of carbon dioxide emissions during the life of the project.
'''

completion = openai.ChatCompletion.create(
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