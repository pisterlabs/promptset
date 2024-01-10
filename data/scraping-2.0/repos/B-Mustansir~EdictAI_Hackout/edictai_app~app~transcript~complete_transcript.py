def lavdeKafunction():
    # Create chunks 

    chunks_list = []

    import os
    import openai
    openai.organization = ""
    openai.api_key = ""

    news = '''
    PM inaugurates 141st International Olympic Committee (IOC) Session in Mumbai "India is eager to host the Olympics in the country. India will leave no stone unturned in the preparation for the successful organization of the Olympics in 2036. This is the dream of the 140 crore Indians" “India is also eager to host the Youth Olympics taking place in the year 2029” “Indians are not just sports lovers, but we also live it” “The sporting legacy of India belongs to the entire world” “In sports, there are no losers, there are only winners and learners” “We are focussing on inclusivity and diversity in sports in India” “IOC Executive Board has recommended including Cricket in the Olympics and we hope to hear positive news soon” Posted On: 14 OCT 2023 9:09PM by PIB Delhi The Prime Minister, Shri Narendra Modi inaugurated the 141st International Olympic Committee (IOC) Session in Mumbai today. The Session provides an opportunity for interaction and knowledge sharing among the various stakeholders related to sports. Addressing the gathering, the Prime Minister underlined the significance of the session taking place in India after 40 years. He also informed the audience about India’s victory in the Cricket World Cup fixture at the world’s largest stadium in Ahmedabad to the roar of cheers. “I congratulate Team Bharat and every Indian on this historic victory”, he said. The Prime Minister emphasized that sports has been a vital part of India’s culture and lifestyle. When you go to the villages of India, the Prime Minister said, one can find that any festival remains incomplete without sports. “Indians are not just sports lovers, but we also live it”, Shri Modi said. He highlighted that the sporting culture is reflected through the thousands year old history of India. Be it the Indus Valley Civilization, the Vedic Period or the era after it, the Prime Minister underlined that India’s sporting legacy has been very prosperous. He informed that scriptures written thousands of years ago mentioned being proficient in 64 genres including sports such as horse riding, swimming, archery wrestling etc. and emphasis was laid on excelling in them. He stated that a ‘Dhanur Veda Samhita’ i.e. a Code for Archery was published to learn the sport of archery which mentions 7 mandatory skills as a prerequisite to learn archery namely Dhanushvan, Chakra, Bhala, Fencing, Dagger, Mace and Wrestling. The Prime Minister presented scientific evidence of this ancient sport legacy of India. He mentioned the Dholavira UNESCO World Heritage site and talked about sports infrastructure in the urban planning of this 5000-year-old city. In the excavation, the Prime Minister said, two stadiums were found, one of them being the oldest and biggest stadium in the world at that time. Similarly, in Rakhigarhi sports-related structures have been found. “This sporting legacy of India belongs to the entire world”, Shri Modi said. Prime Minister Modi said, “There are no losers in sports, only the winners and learners. The language and spirit of sports are universal. Sports is not mere competition. Sports gives humanity an opportunity to expand.” “That is why records are celebrated globally. Sports also strengthens the spirit of ‘Vasudhaiva Kutumbakam’ - One Earth, One Family, One Future”, he added. The Prime Minister also listed recent measures for the development of sports in India. He mentioned Khelo India Games, Khelo India Youth Games, Khelo India Winter Games, Member of Parliament sports competitions and the upcoming Khelo India Para Games. “We are focussing on inclusivity and diversity in sports in India”, the Prime Minister emphasized The Prime Minister credited the efforts of the government for India’s shining performance in the world of sports. He recalled the magnificent performances of many athletes in the last edition of the Olympics and also highlighted India’s best-ever performance in the recently concluded Asian Games and the new records made by young athletes of India in the World University Games.  He underlined that the positive changes are a sign of the rapidly transforming landscape of sports in India. Shri Modi emphasized that India has successfully proved its capability to organize global sports tournaments. He mentioned the recently hosted global tournaments such as the Chess Olympiad which witnessed the participation of 186 countries, the Football Under-17 Women’s World Cup, the Hockey World Cup, the Women’s World Boxing Championship, the Shooting World Cup and the ongoing Cricket World Cup. He also underlined that the nation organizes the largest cricket league in the world every year. The Prime Minister informed that the IOC Executive Board has recommended including Cricket in the Olympics and expressed confidence that the recommendations will be accepted. Underling that global events are an opportunity for India to welcome the world, the Prime Minister emphasized that India is primed to host global events owing to its fast-expanding economy and well-developed infrastructure. He gave the example of the G20 Summit where events were organized in more than 60 cities of the country and said that it is proof of India’s organizing capacity in every sector. The Prime Minister put forth the belief of 140 crore citizens of India “India is eager to host the Olympics in the country. India will leave no stone unturned in the preparation for the successful organization of the Olympics in 2036, this is the dream of the 140 crore Indians”, the Prime Minister said. He emphasized that the nation wishes to fulfill this dream with the support of all stakeholders. “India is also eager to host the Youth Olympics taking place in the year 2029”, Shri Modi remarked and expressed confidence that the IOC will continue lending its support to India. The Prime Minister said that “sports is not just for winning medals but is a medium to win hearts. Sports belongs to all for all. It not only prepares champions but also promotes peace, progress and wellness. Therefore, sports is another medium of uniting the world”. Once again welcoming the delegates, the Prime Minister declared the session open. President of the International Olympic Committee, Mr Thomas Bach and member of International Olympic Committee, Mrs Nita Ambani were present on the occasion among others. Background The IOC session serves as a key meeting of the International Olympic Committee (IOC) members. Important decisions regarding the future of the Olympic games are made at the IOC Sessions. India is hosting the IOC session for the second time after a gap of about 40 years. The IOC's 86th session was held in New Delhi in 1983. The 141st IOC Session, being held in India embodies the nation's dedication to fostering global cooperation, celebrating sporting excellence and furthering the Olympic ideals of friendship, respect, and excellence. It provides an opportunity for interaction and knowledge sharing among the various sports-related stakeholders. The session was also attended by the President of the International Olympic Committee, Mr. Thomas Bach and other members of the IOC, along with prominent Indian sports personalities and representatives from various sports federations, including the Indian Olympic Association.    
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
    import time
    import logging

    speech_config = speechsdk.SpeechConfig(subscription="", region="")
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    speech_config.speech_synthesis_voice_name='en-US-JennyNeural'
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

    combined_chunks = " ".join(sentences)

    speech_synthesis_result = speech_synthesizer.speak_text_async(combined_chunks).get()
    speech_synthesis_stream = speechsdk.AudioDataStream(speech_synthesis_result)
    speech_synthesis_stream.save_to_wav_file("audios/chunk_3.wav")
    print("Audio Produced")

    def text_transcription():
        apnaJson = []
        words = []
        output = []

        audio_filepath = 'audios/chunk_3.wav'  # Replace with your audio file path
        locale = "en-US"  # Change as per requirement

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
    # speech_config = speechsdk.SpeechConfig(subscription="21186bfc40b44f23bdd5d7afe3f19552", region="centralindia")
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
                if (wordlevel_info[k]['word'].strip().lower() == j.strip().lower()):
                    keywordArray.append({'word':j,'start_time':wordlevel_info[k]['start'],'end_time':wordlevel_info[k]['end']})

        print(wordlevel_info[currentStartWord]['start'])
        item = {'chunk': i['sentence'], 'start_time': wordlevel_info[currentStartWord]['start'],
                'end_time': wordlevel_info[currentStartWord+len(i['sentence'].split())-1]['end'], 'keywords': keywordArray}
        output.append(item)
        print(currentStartWord)
        currentStartWord = currentStartWord+len(i['sentence'].split())

    print(output)

lavdeKafunction()