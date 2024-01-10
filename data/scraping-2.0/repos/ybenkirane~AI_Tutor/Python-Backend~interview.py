import openai, os, json, re, random, threading, time, pyaudio, wave, tempfile, audioop, webrtcvad
import numpy as np
from collections import deque
import sounddevice as sd
import soundfile as sf
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
from queue import Queue
from threading import Thread
from elevenlabs import voices, generate, set_api_key

openai.api_key = os.environ["OPENAI_API_KEY"]
set_api_key(os.environ["ELEVENLABS_API_KEY"])
voices = voices()

Industry = "Hedge Fund"
Position = "Quantitative Analyst"
Company = "Company"


# Number of questions per difficulty level - Easy, Medium, Hard, Very Hard
difficultySelectionList_Topics = [1, 0, 0, 0] 
n_topics = sum(difficultySelectionList_Topics)

def remove_control_characters(s):
    return re.sub(r'[\x00-\x1F\x7F]', '', s)

"""
    GENERATING PRE, MAIN, AND POST INTERVIEW QUESTIONS:
        Generating a complete list of questions for the pre, main, and post interview
        stages in JSON formatted blocks to easily extract type, question, and grading rubric. 
        Must be streamed to ensure reasonable response times.
        Each list is saved to a JSON file for later use in which the questions can be split during interview and asked individually
        to save token usage.
"""

def gpt_Q_Gen(question_prompt, title, model):
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
        {"role": "system", "content": f"""You are a top-level {Industry} interviewer. You are interviewing me for the position of {Position}. 
        Consider this prompt: """},
        {"role": "assistant", "content": question_prompt},
        ],
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.5,
        stream=True,
    )

    # Store the chunks in a variable
    chunks = []
    for chunk in response: 
        word = chunk["choices"][0].get("delta", {}).get("content")
        if word is not None:
            chunks.append(word)
            print(word, end='', flush=True)


    # Write to the file only after all chunks have been loaded
    with open(f"{title}.json", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk)
    return ''.join(chunks)
            

def preInterview_Generation():

    # Psychological questions to ask the candidate before the interview.
    # Allow candidate to ask any question about the company (use langchain to extract data from website or internal files.)

    preInterview_Prompt = f"""Generate a list of questions for the start of the interview. These should be simple questions regarding the candidate's cultural background,
    hobbies, interests, risks they've taken, weaknesses, failures, personal icons and heros, and other personal information. Phrase them in a way that relates to the position and the company's culture.
    You may relate them to your own experiences as a top-level {Industry} analyst. Try to empathize with the me (the candidate) and make me feel comfortable.
    Provide list in a JSON format, presenting the "Topic" and "Question".
    Provide 2 questions.
    """
    gpt_Q_Gen(preInterview_Prompt, title="preInterview", model="gpt-4")


def mainInterview_Generation(topic_input):
    from TopicGenerator import topic_generator
    
    Topics = topic_generator(topic_input, n_topics) # A list with b_n random branches
    randomTopics = random.sample(Topics, n_topics)
    print(randomTopics)

    jsonQuestionFormat = """
            {
                "questions": [
                    {
                    "Topic": "Topic",
                    "Difficulty": "Difficulty",
                    "Question": "Question"
                    },
                    {
                    "Topic": "Topic",
                    "Difficulty": "Difficulty",
                    "Question": "Question"
                    }
                    ...
                ]
            }
    """

    mainInterview_TopicPrompt = f"""Generate a list of problems in ascending difficulty relating to these topics: {randomTopics}.
    Provide them in a JSON format, presenting the "Topic", "Difficulty", "Question".
    The difficulty levels are "Easy", "Medium", "Hard", "Very Hard". The "Easy" option should be a fairly straightforward question requiring 1-2 steps to solve or answer. The "Medium" option should begin involving more complex logic. The "Hard" and "Very Hard" options must be extremely difficult, even for a PhD-level graduate. 
    Provide {difficultySelectionList_Topics[0]} Easy question(s), {difficultySelectionList_Topics[1]} Medium Question(s), {difficultySelectionList_Topics[2]} Hard Question(s), and {difficultySelectionList_Topics[3]} Very Hard question(s). 
    There should be a total of {n_topics} questions. Make sure these are intricate and well-thought-out. 
    You must follow the format: """ + jsonQuestionFormat
    
    gpt_Q_Gen(mainInterview_TopicPrompt, title="mainInterviewTopic", model="gpt-4")

def postInterview_Generation():
    # Just thank the user for completing the interview. No prompt probably required. Pre-recorded audio file.

    postInterview_Prompt = f"""Generate a list of brain teasers in ascending difficulty relating to these topics: 
    (logic, puzzles, probability, statistics, financial, trading, game theory, estimation, mathematics, or physics). 
    Provide them in a JSON format, presenting the "Topic", "Difficulty", "Question".
    The difficulty may be stated as "Easy", "Medium", "Hard", "Very Hard". Provide approximately 3 questions per difficulty level.
    """
    gpt_Q_Gen(postInterview_Prompt, title="postInterview", model="gpt-4")

"""
Solution and Rubric Generation:
"""

def rubricGenerator(iteration):
    with open(f"mainInterviewTopic.json", "r") as f:
        dataQuestion = json.load(f)
    
    question = dataQuestion["questions"][iteration]["Question"]

    jsonSolutionFormat = """
                {
                    "solution": [
                        {
                        "Question": "Question Number",
                        "Answer": " Complete Answer to the first question with structured reasoning and detailed steps."
                        }
                    ]
                }
        """
    # FORGOT WE WERE ITERATING THROUGH THE QUESTIONS IN THE JSON FILE --> NEED TO EITHER BE OK WITH MULTIPLE OBJECT SOLUTION IN ONE JSON FILE OR CREATE MULTIPLE JSON SOLUTION FILES..
    solution_Prompt = f"""Generate a solution to the following question:{question}. Provide a highly detailed solution that covers all possible cases, discusses the theory behind 
    the solution and the reasoning behind each step. It must be clear and consise.
    Return the solution in a JSON format, presenting the "Question Number {iteration}", "Solution". 
    You must folow this format: """ + jsonSolutionFormat
    gpt_Q_Gen(solution_Prompt, title=f"solution{iteration}", model="gpt-4")

    with open(f"solution{iteration}.json", "r") as f:
        file_content = f.read()
        cleaned_content = remove_control_characters(file_content)
        # escaped_content = cleaned_content.replace('\\', '\\\\')
        dataSolution = json.loads(cleaned_content)

    solution = dataSolution["solution"][0]["Answer"]

    # The solution will often contain lots of \n characters. These must be parsed out. 
    jsonRubricFormat = """
                {
                    "rubric": [
                        {
                        "Question": "Question Number",
                        "Rubric": " Complete Rubric to the question based on the example solution provided."
                        }
                    ]
                }
        """

    # Rubric Generation
    rubric_Prompt = f"""Generate a rubric on how to grade to the following problem: {question}. Make use of on the generated example solution: {solution}. 
    The rubric must be highly critical and structured, such that the interviewer may easily identify flaws in the candidate's thought process and solution. 
    It is critical that the rubric presents exactly how each point can be acquired and what differentiates each point acquired. 
    The marks / points must be integer values that sum up to a total score. 
    A correct and structured thought process is more important than achieving the correct final solution. The rubric will be used to grade the candidate's answer.
    This rubric must identify how each part of the solution will be graded, how to allocate marks, and how to identify the quality of the answer.
    Return the solution in a JSON format, presenting the "Question Number {iteration}", "Rubric". 
    You must folow this format: """ + jsonRubricFormat
    gpt_Q_Gen(rubric_Prompt, title=f"rubric{iteration}", model="gpt-4")


"""
Grading Candidate's Solution:
"""

def gradeCandidate(answer, iteration):
    with open(f"rubric{iteration}.json", "r") as f:
        file_content = f.read()
        cleaned_content = remove_control_characters(file_content)
        dataRubric = json.loads(cleaned_content)
    
    with open(f"mainInterviewTopic.json", "r") as f:
        dataQuestion = json.load(f)
    
    question = dataQuestion["questions"][iteration]["Question"]
    rubric = dataRubric["rubric"][0]["Rubric"]

    jsonGradingFormat = """
                {
                    "grade": [
                        {
                        "Solution": "Solution Number",
                        "Remarks": "Complete Remarks to the solution provided by the candidate. Going over which points were achieved and which were not.",
                        "Score": "(Achieved Score) / (Maximum Score)"
                        }
                    ]
                }
    """

    grading_Prompt = f"""The question asked to the candidate was: {question}. 
    Based on the rubric: {rubric}, please grade the candidate's solution: {answer}. 
    Carefully analyze the candidate's thought process and solution. Describe and misteps that they took and praise any keen insights.
    Return the grading output in a JSON format, presenting the "Solution Number {iteration}", "Remarks", and "Score". 
    You must folow this format: """ + jsonGradingFormat

    gpt_Q_Gen(grading_Prompt, title=f"score{iteration}", model="gpt-4")    

    #Now we extract the score from the JSON output.
    with open(f"score{iteration}.json", "r") as f:
        file_content = f.read()
        cleaned_content = remove_control_characters(file_content)
        dataScore = json.loads(cleaned_content)
        
    feedback = dataScore["grade"][0]["Remarks"]
    score = dataScore["grade"][0]["Score"]
    return feedback, score


"""
Audio Transcription:
"""

global RATE, CHUNK_DURATION_MS
CHUNK_DURATION_MS = 30 # In milliseconds
RATE = 48000    # Sampling rate in Hz
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000) # Number of samples in each chunk
FORMAT = pyaudio.paInt16  # Audio format (bytes per sample)
CHANNELS = 1   # Mono audio
AUDIO_INTENSITY_THRESHOLD = 500  # Experiment with different values to get the best performance
ROLLING_WINDOW_SIZE = 6 # Number of chunks to smooth over when calculating average audio intensity
vad = webrtcvad.Vad(3)  # Change the aggressiveness level (0 to 3)
OVERLAP_CHUNKS = 2  # overlapping 2 to 3 chunks (i.e., 60-90 ms) should be sufficient to maintain context without causing repeated words
audio = pyaudio.PyAudio() 
transcription_lock = threading.Lock() # Used to synchronize callbacks with the main thread

def transcribe(audio_file_path): 
    with open(audio_file_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe(file = audio_file,
                                             model = "whisper-1",
                                             response_format = "text",
                                             language = "en", #Supports multilingual transcription, but adding this parameter will improve performance
                                            ) 
    return transcript

def transcribe_audio(speech_frames, transcript, task_order, current_task):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        with wave.open(temp_file.name, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(speech_frames))

        transcript_chunk = transcribe(temp_file.name)

        with transcription_lock:
            while task_order[0] != current_task:
                transcription_lock.release()
                time.sleep(0.1)
                transcription_lock.acquire()

            current_transcript_length = len("".join(transcript))
            transcript_chunk = transcript_chunk.strip()  # Remove extra spaces before appending
            transcript.append(transcript_chunk + " ")
            task_order.pop(0)

            # Print only the new words added to the transcript
            print("".join(transcript)[current_transcript_length:], end="", flush=True)

def audio_intensity(chunk):
    return audioop.rms(chunk, 2)

def record_audio(question, n):

    transcript = []
    task_order = []
    task_number = 0    

    stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)

    print("Start recording your solution. Clearly walk through your entire thought process. Say 'I am done' when finished.")

    speech_frames = []
    audio_intensities = deque(maxlen=ROLLING_WINDOW_SIZE)
    chunk = stream.read(CHUNK_SIZE)
    audio_intensities.append(audio_intensity(chunk))
    speech_duration_ms = 0  # Add a variable to track the total speech duration collected
    silent_frame_count = 0

    hint_counter = 0
    max_hints = 2 #Maximum number of hints allowed per question

    while True:
        is_speech = False

        try:

            if ("i am done" in "".join(transcript).lower()) or ("i'm done" in "".join(transcript).lower()) or ("i am finished" in "".join(transcript).lower()) or ("i'm finished" in "".join(transcript).lower()):
                print("You have submitted your solution to this problem. Please wait...")
                break
            elif (("need a hint" in "".join(transcript).lower())) and (hint_counter <= max_hints):
                #  or ("need a tip") in "".join(transcript).lower()
                hint_prompt = """Carefully analyze the question below. The candidate is slightly lost on how to solve the problem. 
                Provide a hint to the candidate regarding the question; it must hold new insight while not revealing the solution. 
                Do not provide the solution, only a hint. The question is: """ + question
                hint = gpt_Q_Gen(hint_prompt, title=f"hint{n}", model="gpt-3.5-turbo")
                ttsGeneration(hint)
                transcript_str = "".join(transcript).lower()
                transcript_str = transcript_str.replace("i need a hint", "")
                transcript = transcript_str.split()
                hint_counter += 1
            elif (("need a hint" in "".join(transcript).lower()) or ("need a tip") in "".join(transcript).lower()) and (hint_counter > max_hints):
                print("You have reached the maximum number of hints. Please continue with your solution.")

            chunk = stream.read(CHUNK_SIZE)
            audio_intensities.append(audio_intensity(chunk))
            avg_intensity = np.mean(audio_intensities)

            if avg_intensity > AUDIO_INTENSITY_THRESHOLD:
                try:
                    is_speech = vad.is_speech(chunk, RATE)
                except Exception:
                    print("Error processing frame, skipping...")
                    continue

            if is_speech:
                # print("Speech detected")  # Debug information
                speech_frames.append(chunk)
                speech_duration_ms += CHUNK_DURATION_MS  # Increment the speech duration
                # print(speech_duration_ms)
                silent_frame_count = 0  # Reset the silent frame count when speech is detected

                if speech_duration_ms >= 10000:  # Check if the speech duration exceeds 1 second
                    # Start a new thread for the transcription process
                    transcription_thread = threading.Thread(target=transcribe_audio, args=(speech_frames.copy(), transcript, task_order, task_number))
                    transcription_thread.start()

                    with transcription_lock:
                        task_order.append(task_number)

                    task_number += 1
                    
                    # Reset the speech_frames buffer
                    overlap_frames = speech_frames[-OVERLAP_CHUNKS:]  # Save the overlapping frames
                    speech_frames = overlap_frames  # Start the new speech_frames list with the overlapping frames
                    speech_duration_ms = CHUNK_DURATION_MS * len(speech_frames)  # Update the speech duration accordingly

            else:
                silent_frame_count += 1

            # Check if the silence duration reaches one second
            silence_duration_ms = silent_frame_count * CHUNK_DURATION_MS
            try: 
                if silence_duration_ms >= 5000 and len(speech_frames)*CHUNK_DURATION_MS > 150:
                    # Start a new thread for the transcription process
                    transcription_thread = threading.Thread(target=transcribe_audio, args=(speech_frames.copy(), transcript, task_order, task_number))
                    transcription_thread.start()

                    with transcription_lock:
                        task_order.append(task_number)

                    task_number += 1

                    # Reset the speech_frames buffer
                    speech_frames = []
                    silent_frame_count = 0

            except Exception:
                print("Error processing silence threshold limit frame, skipping...")

        except KeyboardInterrupt:
            print("Stopping... 1")
            break

    print("Stopping... 2")

    # # Wait for all transcription threads to finish
    # for thread in threading.enumerate():
    #     if thread is not threading.current_thread():
    #         thread.join()

    print("Recording finished, processing...")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the transcript to a text file
    transcript_str = ' '.join(transcript)
    with open(f'transcript{n}.txt', 'w') as file:
        file.write(transcript_str)

    return transcript

def ttsGeneration(interview_text: str):
    # Text to speech to a file
    fd, temp_file_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    # Use ElevenLabs API to generate audio
    generated_audio = generate(
        text=interview_text,
        voice=voices[12],  # Choose the desired voice from the list : current default is 12 (Casper)
    )

    # Save the generated audio to a temporary file
    with open(temp_file_path, "wb") as f:
        f.write(generated_audio)

    data, samplerate = sf.read(temp_file_path)
    sd.play(data, samplerate)
    sd.wait()

    return temp_file_path

def grade_responses(q, results):
    while True:
        solution, n = q.get()
        # Wait until the rubric file is created
        while not os.path.isfile(f'rubric{n}.json'):
            time.sleep(1)  # Wait for 1 second before checking again
        feedback, score = gradeCandidate(solution, n)
        with open(f"candidateSolution{n}.txt", "w", encoding="utf-8") as f:
            f.write('\n'.join(solution))
        results.append((n, feedback, score))
        q.task_done()

def candidateInterview(q):
    for n in range(0, n_topics):
        with open(f"mainInterviewTopic.json", "r") as f:
            data = json.load(f)
        question = data["questions"][n]["Question"]
        print(f'{Fore.GREEN} {question} {Style.RESET_ALL}')
        ttsGeneration(question)
        solution = record_audio(question, n) # Collect the user's solution via audio recording
        q.put((solution, n)) # Add the solution to the queue.

def rubric_generator_thread(question_number):
    rubricGenerator(question_number)

def main():
    from pdfGenerator import pdfGeneration

    topic = input("Type in the Topic of the Interview: ")
    mainInterview_Generation(topic)

    # Create a list to hold all rubric generation threads
    rubric_threads = []

    # Spawn a new thread for each rubric generation task
    for question_number in range(n_topics):
        thread = threading.Thread(target=rubric_generator_thread, args=(question_number,))
        thread.start()
        rubric_threads.append(thread)

    q = Queue()
    results = []
    grading_thread = Thread(target=grade_responses, args=(q, results))
    grading_thread.start()
    candidateInterview(q)
    q.join()  # Wait for all solutions to be graded
    
    # Wait for all rubric generation tasks to complete
    for thread in rubric_threads:
        thread.join()

    # Sort results by question number and print them
    results.sort()
    for n, feedback, score in results:
        print(feedback)
        ttsGeneration(feedback)
        print(f"Your score to question {n} is: {score}")

    pdfGeneration()
    exit()

if __name__ == "__main__":
    main()

# Fix Latex equation in chat completion results --> PDF Generation
# Fix any parsing and formating potential issues
# Handle any possible Exceptions
# Repeat GPT Generation in case an error arises or JSON file in incorrect format. 