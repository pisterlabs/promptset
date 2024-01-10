def lavdeKafunction():
    # Create chunks 

    import json
    chunks_list = []
    from Keyword_json import subject_json

    import os
    import openai
    openai.organization = ""
    openai.api_key = ""

    news = '''
    Bharat Mata Ki Jai! The popular and young Chief Minister of Uttarakhand Bhai Pushkar Singh Dhami ji, Union Minister Shri Ajay Bhatt ji, former Chief Minister Ramesh Pokhriyal Nishank ji, State President of Bharatiya Janata Party Mahendra Bhatt ji, Ministers of Uttarakhand Government, all MPs, MLAs, other dignitaries and my dear family members of Devbhoomi, greetings to all of you! Today Uttarakhand has done wonders. Perhaps no one has had the privilege of witnessing such a scene before. Wherever I went across Uttarakhand since morning, I was showered with immense love and blessings; It seemed as if the river (Ganga) of love was flowing. I salute this land of spirituality and unmatched bravery. I particularly salute the brave mothers. When the battle cry of "Jai Badri-Vishal" is raised in Badrinath Dham, the spirit and vigour of the bravehearts of Garhwal Rifles rises. When the bells of the Kalika temple of Gangolihat echo with the battle cry of "Jai Mahakali", indomitable courage starts flowing among the heroes of the Kumaon Regiment. Here we have the splendour of Bageshwar in Manaskhand, Baijnath, Nanda Devi, Golu Devta, Purnagiri, Kasar Devi, Kainchi Dham, Katarmal, Nanakmatta, Reetha sahib and countless pilgrimage sites. We possess a rich heritage.
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
    for sentence in sentences:
        keywords = []
        # kw_model = KeyBERT()
        # extracted_keywords = kw_model.extract_keywords(sentence,keyphrase_ngram_range=(1, 1))
        # keywords = []
        # if len(extracted_keywords)>3:
        #     extracted_keywords = extracted_keywords[0:3]
        # for key in extracted_keywords: 
        #     keywords.append(key[0])
        output =  subject_json(sentence)
        print(output)
        start_index = output.index('{')
        end_index = output.rindex('}')
        data = output[start_index:end_index+1]
        keywords_str = json.loads(data)
        print(keywords_str.get('key phrase or entity'))
        for i in keywords_str.get('key phrase or entity').split(','):
            if i!=None and i!='' and i!="None":
                keywords.append(i)
        if keywords_str.get('Sub-Subject')!=None and keywords_str.get('Sub-Subject')!='' and keywords_str.get('Sub-Subject')!="None":
            keywords.append(keywords_str.get('Sub-Subject'))
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
    import time
    import logging

    speech_config = speechsdk.SpeechConfig(subscription="", region="")
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    speech_config.speech_synthesis_voice_name='en-IN-NeerjaNeural'
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

    combined_chunks = " ".join(sentences)

    speech_synthesis_result = speech_synthesizer.speak_text_async(combined_chunks).get()
    speech_synthesis_stream = speechsdk.AudioDataStream(speech_synthesis_result)
    speech_synthesis_stream.save_to_wav_file("audios/chunk_4.wav")
    print("Audio Produced")

    def text_transcription():
        apnaJson = []
        words = []
        output = []

        audio_filepath = 'audios/chunk_4.wav'  # Replace with your audio file path
        locale = "en-IN"  # Change as per requirement

        # logger.debug(audio_filepath)
        audio_config = speechsdk.audio.AudioConfig(filename=audio_filepath)
        speech_config = speechsdk.SpeechConfig(subscription="", region="")
        speech_config.request_word_level_timestamps()
        speech_config.speech_recognition_language = locale
        speech_config.output_format = speechsdk.OutputFormat(1)

        # Creates a recognizer with the given settings
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        # Variable to monitor status
        done = False

        # Service callback for recognition text
        def parse_azure_result(evt):
            import json
            response = json.loads(evt.result.json)
            apnaJson.append(response)
            # logger.debug(evt)

        # Service callback that stops continuous recognition upon receiving an event evt
        def stop_cb(evt):
            print('CLOSING on {}'.format(evt))
            speech_recognizer.stop_continuous_recognition()
            nonlocal done
            done = True

        # Connect callbacks to the events fired by the speech recognizer
        # speech_recognizer.recognizing.connect(lambda evt: logger.debug('RECOGNIZING: {}'.format(evt)))
        speech_recognizer.recognized.connect(parse_azure_result)
        # speech_recognizer.session_started.connect(lambda evt: logger.debug('SESSION STARTED: {}'.format(evt)))
        # speech_recognizer.session_stopped.connect(lambda evt: logger.debug('SESSION STOPPED {}'.format(evt)))
        # speech_recognizer.canceled.connect(lambda evt: logger.debug('CANCELED {}'.format(evt)))
        # Stop continuous recognition on either session stopped or canceled events
        speech_recognizer.session_stopped.connect(stop_cb)
        speech_recognizer.canceled.connect(stop_cb)

        # Start continuous speech recognition
        # logger.debug("Initiating speech to text")
        speech_recognizer.start_continuous_recognition()
        while not done:
            time.sleep(.5)

        # Process apnaJson to create 'output' and 'words'
        for jid in apnaJson:
            output.append({'chunk': jid['DisplayText'], 'start': (jid['Offset'] / 10000000), 'end': ((jid['Duration'] + jid['Offset']) / 10000000)})

        for jid in apnaJson:
            for chunk_words in jid['NBest'][0]['Words']:
                words.append({'word': chunk_words['Word'], 'start': chunk_words['Offset'] / 10000000, 'end': ((chunk_words['Duration'] + chunk_words['Offset']) / 10000000)})

        return (words)

    # import os
    # import azure.cognitiveservices.speech as speechsdk
    # speech_config = speechsdk.SpeechConfig(subscription="", region="")
    # audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    # speech_config.speech_synthesis_voice_name='en-US-JennyNeural'
    # speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

    # combined_chunks = " ".join(sentences)

    # speech_synthesis_result = speech_synthesizer.speak_text_async(combined_chunks).get()
    # speech_synthesis_stream = speechsdk.AudioDataStream(speech_synthesis_result)
    # speech_synthesis_stream.save_to_wav_file("chunk.wav")

    # if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
    #     print("Speech synthesized for text [{}]".format(news))
    # elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
    #     cancellation_details = speech_synthesis_result.cancellation_details
    #     print("Speech synthesis canceled: {}".format(cancellation_details.reason))
    #     if cancellation_details.reason == speechsdk.CancellationReason.Error:
    #         if cancellation_details.error_details:
    #             print("Error details: {}".format(cancellation_details.error_details))
    #             print("Did you set the speech resource key and region values?")

    # from faster_whisper import WhisperModel
    # model_size = "medium"
    # model = WhisperModel(model_size)

    # segments, info = model.transcribe("chunk.wav", word_timestamps=True)
    # segments = list(segments)
    # for segment in segments:
    #     for word in segment.words:
    #         print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))

    # wordlevel_info = []

    # for segment in segments:
    #     for word in segment.words:
    #       wordlevel_info.append({'word':word.word,'start':word.start,'end':word.end})

    # print(wordlevel_info)

    wordlevel_info = text_transcription()
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
                j_list = j.split()
                if (wordlevel_info[k]['word'].strip().lower() == j_list[0].strip().lower()):
                    key_phrase_start_time = wordlevel_info[k]['start']
                if (wordlevel_info[k]['word'].strip().lower() == j_list[-1].strip().lower()):
                    key_phrase_end_time = wordlevel_info[k]['end']
                    keywordArray.append({'word':j,'start_time':key_phrase_start_time,'end_time':key_phrase_end_time})

        print(wordlevel_info[currentStartWord]['start'])
        item = {'chunk': i['sentence'], 'start_time': wordlevel_info[currentStartWord]['start'],
                'end_time': wordlevel_info[currentStartWord+len(i['sentence'].split())-1]['end'], 'keywords': keywordArray}
        output.append(item)
        print(currentStartWord)
        currentStartWord = currentStartWord+len(i['sentence'].split())

    print(output)

lavdeKafunction()