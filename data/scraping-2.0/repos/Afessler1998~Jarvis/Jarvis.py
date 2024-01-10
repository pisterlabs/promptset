from Audio_Processing import process_audio
from datetime import datetime
from multiprocessing import Process, Queue, Event
from OpenAI_Interface import OpenAI_Interface
from Pinecone_Interface import Pinecone_Interface
from Speech_Interface import record_audio, wav_to_audio, speech_to_text, text_to_speech


def speech_recognition_process(user_messages_queue,
                               audio_queue,
                               pause_listening_event,
                               stop_event):
    
    error_twice_sequentially = False
    while not stop_event.is_set():
        if pause_listening_event.is_set():
            continue
        try:
            audio = record_audio()
            if pause_listening_event.is_set():
                continue
            if audio is not None:
                audio_wav = audio.get_wav_data()
                with open("system_files/user_message.wav", "wb") as f:
                    f.write(audio_wav)
                    process_audio("system_files/user_message.wav")
                    audio = wav_to_audio("system_files/user_message.wav")
                # Process audio before determining if keyword is present to maximize responsiveness
                # Uneeded results will be discarded in the message handling process
                audio_queue.put(audio)
                user_messages_queue.put(audio)

        except Exception as e:
            print("ERROR RECORDING AUDIO: ", e)
            if error_twice_sequentially:
                stop_event.set()
                print("Stopping audio recording process due to repeated errors.")
                break
            else:
                error_twice_sequentially = True


def voice_analysis_process(audio_queue,
                           speaker_recognized_queue,
                           emotion_recognized_queue,
                           stop_event):
    
    error_twice_sequentially = False
    while not stop_event.is_set():
        try:
            audio = audio_queue.get()
            print("Audio received for analysis.")
            
            # speaker recognition
            speaker_recognized_queue.put("Alec")
            print("Speaker recognized.")

            # emotion recognition
            emotion_recognized_queue.put("Neutral")
            print("Emotion recognized.")

        except Exception as e:
            print("ERROR PROCESSING AUDIO: ", e)
            if error_twice_sequentially:
                stop_event.set()
                print("Stopping audio processing process due to repeated errors.")
                break
            else:
                error_twice_sequentially = True


def message_handling_process(user_messages_queue,
                             speaker_recognized_queue,
                             emotion_recognized_queue,
                             jarvis_messages_queue,
                             db_upsert_queue,
                             db_query_queue,
                             db_response_queue,
                             pause_listening_event,
                             stop_event):
    
    error_twice_sequentially = False
    receiving_response_error_twice_sequentially = False
    OpenAI = OpenAI_Interface(model="gpt-4-1106-preview",
                              temperature=0,
                              stream=True,
                              db_upsert_queue=db_upsert_queue,
                              jarvis_messages_queue=jarvis_messages_queue)

    while not stop_event.is_set():
        try:
            audio = user_messages_queue.get()
            text = speech_to_text(audio)
            text_lower = text.lower()

            if text:
                if "stop listening" in text_lower:
                    print("Shutting down Jarvis.")
                    stop_event.set()
                    break
                elif "jarvis" in text_lower:
                    try:
                        pause_listening_event.set()
                        user = speaker_recognized_queue.get()
                        print(f"User: {user}\n")
                        timestamp = datetime.now().isoformat()
                        db_query_queue.put((f"User: {user}\n"
                                            f"Timestamp: {timestamp}\n"
                                            f"User message: {text}\n"))
                        db_results = db_response_queue.get()
                        user_emotion = emotion_recognized_queue.get()
                        print(f"User emotion: {user_emotion}\n")

                        prompt = OpenAI.make_prompt(user, user_emotion, timestamp, text, db_results)
                        print(prompt)
                        OpenAI.message("user", prompt, OpenAI.tools)
                        continue

                    except Exception as e:
                        print("ERROR RECEIVING RESPONSE: ", e)
                        if receiving_response_error_twice_sequentially:
                            stop_event.set()
                            print("Stopping due to repeated errors in receiving response.")
                            break
                        else:
                            receiving_response_error_twice_sequentially = True

            # Remove irrelevant audio analysis results
            if not speaker_recognized_queue.empty() and not emotion_recognized_queue.empty():
                speaker_recognized_queue.get()
                emotion_recognized_queue.get()
                print("Cleared irrelevant audio analysis results.")

        except Exception as e:
            print("ERROR IN AUDIO PROCESSING: ", e)
            if error_twice_sequentially:
                stop_event.set()
                print("Stopping due to repeated errors in audio processing.")
                break
            else:
                error_twice_sequentially = True

    OpenAI.save_messages("system_files/messages.json", "Alec")


def text_to_speech_process(jarvis_messages_queue,
                           pause_listening_event,
                           stop_event):
    
    text_to_speech("Systems online, Sir.")
    error_twice_sequentially = False
    while not stop_event.is_set():
        try:
            sentence = jarvis_messages_queue.get()
            print(f"Speaking: {sentence}\n") 
            text_to_speech(sentence)
            pause_listening_event.clear()

        except Exception as e:
            print("ERROR IN TTS: ", e)
            if error_twice_sequentially:
                stop_event.set()
                print("Stopping TTS process due to repeated errors.")
                break
            else:
                error_twice_sequentially = True


def database_interaction_process(db_upsert_queue,
                                 db_query_queue,
                                 db_response_queue,
                                 stop_event):
    
    pinecone_interface = Pinecone_Interface()    
    while not stop_event.is_set():
        try:
            while not db_upsert_queue.empty():
                db_info = db_upsert_queue.get()
                print(f"Upserting to index: {db_info}\n")
                pinecone_interface.upsert_to_index(db_info)

        except Exception as e:
            print("ERROR UPSERTING TO INDEX: ", e)

        try:
            while not db_query_queue.empty():
                query_text = db_query_queue.get()
                results = pinecone_interface.query_index(query_text)
                db_response_queue.put(results)

        except Exception as e:
            print("ERROR QUERYING INDEX: ", e)


if __name__ == "__main__":
    stop_event = Event()
    pause_listening_event = Event()
    user_messages_queue = Queue()
    audio_queue = Queue()
    speaker_recognized_queue = Queue()
    emotion_recognized_queue = Queue()
    jarvis_messages_queue = Queue()
    db_upsert_queue = Queue()
    db_query_queue = Queue()
    db_response_queue = Queue()

    speech_recognition = Process(target=speech_recognition_process, args=(user_messages_queue, audio_queue, pause_listening_event, stop_event))
    voice_analysis = Process(target=voice_analysis_process, args=(audio_queue, speaker_recognized_queue, emotion_recognized_queue, stop_event))
    message_handling = Process(target=message_handling_process, args=(user_messages_queue, speaker_recognized_queue, emotion_recognized_queue, jarvis_messages_queue, db_upsert_queue, db_query_queue, db_response_queue, pause_listening_event, stop_event))
    text_to_speech = Process(target=text_to_speech_process, args=(jarvis_messages_queue, pause_listening_event, stop_event))
    database_interaction = Process(target=database_interaction_process, args=(db_upsert_queue, db_query_queue, db_response_queue, stop_event))

    speech_recognition.start()
    voice_analysis.start()
    message_handling.start()
    text_to_speech.start()
    database_interaction.start()

    speech_recognition.join()
    voice_analysis.join()
    message_handling.join()
    text_to_speech.join()
    database_interaction.join()