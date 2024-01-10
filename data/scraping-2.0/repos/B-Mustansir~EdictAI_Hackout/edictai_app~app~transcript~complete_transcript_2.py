# Create chunks 

chunks_list = []

import os
import openai
openai.organization = "org-5LY5AiUTjRELf7YC1UtBfo0j"
openai.api_key = "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"

news = '''
PM thanks artists for rendition of his Garba song Posted On: 14 OCT 2023 11:57AM by PIB Delhi The Prime Minister, Shri Narendra Modi today thanked artists Dhvani Bhanushali, Tanishk Bagchi and team of Jjust  Music for musical, rendition of a Garba that he had penned years ago. He also informed that he will share a new Garba during the upcoming Navratri. Shri Narendra Modi posted on X : "Thank you @dhvanivinod, Tanishk Bagchi and the team of @Jjust_Music for this lovely rendition of a Garba I had penned years ago! It does bring back many memories. I have not written for many years now but I did manage to write a new Garba over the last few days, which I will share during Navratri. #SoulfulGarba"'''

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
print()

# Separating chunks
import re
chunks = completion.choices[0].message.content
sentences = re.split(r"<m>|\\n|\n|</m>",chunks)

sentences = [sentence.strip() for sentence in sentences]
sentences = [sentence for sentence in sentences if sentence]

print(sentences)
print()

# Creating keywords for sentences
from keybert import KeyBERT

for sentence in sentences:
    kw_model = KeyBERT()
    extracted_keywords = kw_model.extract_keywords(sentence,keyphrase_ngram_range=(1, 1))
    keywords = []
    if len(extracted_keywords)>3:
        extracted_keywords = extracted_keywords[0:3]
    for key in extracted_keywords: 
        keywords.append(key[0])
    print(keywords)
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     temperature=0.25,
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": '''
    #             You will be provided with a block of text, and your task is to extract a list of keywords from it.
    #             Note: Keywords extracted would be used as a query to search for images on search engines.
    #             Please avoid unnecessary details or tangential points.
    #             '''
    #         },
    #         {
    #             "role": "user",
    #             "content": sentence
    #         }
    #     ]
    # )
    # keywords_str = response['choices'][0]['message']['content']
    # print (keywords_str)

    chunks_list.append({'sentence':sentence, 'keywords': keywords})

print()
print(chunks_list)
print()

import os
import azure.cognitiveservices.speech as speechsdk
speech_config = speechsdk.SpeechConfig(subscription="21186bfc40b44f23bdd5d7afe3f19552", region="centralindia")
audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
speech_config.speech_synthesis_voice_name='en-US-JennyNeural'
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

combined_chunks = " ".join(sentences)

speech_synthesis_result = speech_synthesizer.speak_text_async(combined_chunks).get()
speech_synthesis_stream = speechsdk.AudioDataStream(speech_synthesis_result)
speech_synthesis_stream.save_to_wav_file("chunk_2.wav")

if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
    print("Speech synthesized for text [{}]".format(news))
elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
    cancellation_details = speech_synthesis_result.cancellation_details
    print("Speech synthesis canceled: {}".format(cancellation_details.reason))
    if cancellation_details.reason == speechsdk.CancellationReason.Error:
        if cancellation_details.error_details:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")

from faster_whisper import WhisperModel
model_size = "small"
model = WhisperModel(model_size)

segments, info = model.transcribe("chunk.wav", word_timestamps=True)
segments = list(segments)
for segment in segments:
    for word in segment.words:
        print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))

wordlevel_info = []

for segment in segments:
    for word in segment.words:
      wordlevel_info.append({'word':word.word.strip().lower(),'start':word.start,'end':word.end})

print(wordlevel_info)

# JSON Converter 

# output = []
# item={'chunk':chunks_list[0]['sentence'],'start_time':wordlevel_info[0]['start'],'end_time':wordlevel_info[0+len(chunks_list[0]['sentence'].split())]['end'],'keywords':chunks_list[0]['keywords']}
# print(item)
# # print(wordlevel_info[len(chunks_list[0]['sentence'])]['end'])

# for i in range(0,len(chunks_list[0]['sentence'].split())):
#     if(wordlevel_info[i]['word']==' Good'):
#         output.append(wordlevel_info[i]['word'])
#     else:continue

output = []
currentStartWord = 0
for i in chunks_list:
    keywordArray = []
    for j in i['keywords']:
        for k in range(currentStartWord, currentStartWord+len(i['sentence'].split())):
            if (wordlevel_info[k]['word'] == j):
                keywordArray.append({'word':j,'start_time':wordlevel_info[k]['start'],'end_time':wordlevel_info[k]['end']})

    print(wordlevel_info[currentStartWord]['start'])
    item = {'chunk': i['sentence'], 'start_time': wordlevel_info[currentStartWord]['start'],
            'end_time': wordlevel_info[currentStartWord+len(i['sentence'].split())-1]['end'], 'keywords': keywordArray}
    output.append(item)
    print(currentStartWord)
    currentStartWord = currentStartWord+len(i['sentence'].split())

print(output)