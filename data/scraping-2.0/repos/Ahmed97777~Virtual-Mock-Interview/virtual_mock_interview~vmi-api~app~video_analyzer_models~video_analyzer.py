import cv2
import os
import openai
import subprocess
import multiprocessing as mp
from app.video_analyzer_models.modules.facial_model import FacialModel
from app.video_analyzer_models.modules.voice_model import VoiceModel
from app import app
import pickle

class VideoAnalyzer:
    @staticmethod
    def analyze_video(interview_id,video_id, question, DEBUG=False):
        
        openai.api_key = app.config['OPENAI_API_KEY']
        voiceModel = VoiceModel()
        result_dict = {}
        original_video_path = '{}/{}/{}'.format(app.config["UPLOAD_FOLDER"], interview_id, video_id)
        output_video_path = '{}/{}/{}'.format(app.config["DOWNLOAD_FOLDER"], interview_id, video_id)
        
        # using ffmpeg to convert the video to mp4 and seperate the audio from the video
        process1 = subprocess.call('ffmpeg -fflags +genpts -i {}.webm -r 30 {}-input.mp4'.format(original_video_path,output_video_path),shell=True)
        process2 = subprocess.call('ffmpeg -i {}-input.mp4 -q:a 0 -map a {}.wav'.format(output_video_path,output_video_path),shell=True)
        
        # using the voice model to analyze the audio
        result_dict.update(voiceModel.voiceModel(output_video_path , DEBUG))
        # delete the voice model object to free up memory
        del voiceModel
       
        # create a process to run the facial model
        ctx = mp.get_context('spawn')
        queue = ctx.Queue()
        p1 = ctx.Process(target=VideoAnalyzer.analyze_vid_process, args=(queue ,output_video_path, DEBUG))
        p1.start()
        # get the result from the facial model process
        result_dict.update(queue.get())
        # terminate the facial model process and free up memory
        p1.terminate()
        # get the gpt response from openai and add it to the result_dict
        try:
            gpt_response = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo",
                messages = [
                    {"role": "system", "content": "You are a helpful interviewer that provides feedback on the interviewee's answer directly to the interviewee. Mention the interviewee's sentences structure , also mention whether the words they used are professional or not, also comment on the energy of the interviewee during answering the question, lastly, provide examples whenever possible whenever there is a window for improvment in the interviewee's speech, sentences structure and words. You must not ask questions."},
                    {"role": "user", "content":'I got asked question: `{}`, and I answered `{}`, mostly my energy was {} during the question.'.format(question, result_dict['text'], result_dict['most_energy'])},
                ]
            )
            print(gpt_response['choices'][0]['message']['content'])
            result_dict['gpt_response'] = gpt_response['choices'][0]['message']['content']
        except:
            result_dict['gpt_response'] = "Sorry, OpenAI is not working right now, please try again later."
        # merge the audio and video together
        process3 = subprocess.call('ffmpeg -i {}-temp.mp4 -i {}.wav -c:v libx264 -c:a aac -strict -2 -map 0:v:0 -map 1:a:0 {}-temp-audio.mp4'.format(output_video_path, output_video_path, output_video_path),shell=True)
        # delete the temp files
        #os.remove('{}-temp.mp4'.format(path_to_video_id))
        os.rename('{}-temp-audio.mp4'.format(output_video_path), '{}.mp4'.format(output_video_path))

        # create a pickle file to store the result dictionary
        with open('{}.pkl'.format(output_video_path), 'wb+') as f:
            pickle.dump(result_dict, f, pickle.HIGHEST_PROTOCOL)
        return

    @staticmethod
    def analyze_vid_process(queue,video_path, DEBUG):
        # load facial model here instead of init to utilize gpu usage
        facialModel = FacialModel()
        # using opencv library to process the video and send the frames to the facial model
        iris_pos_per_frame = []
        facial_emotion_per_frame = []
        facial_emotion_prob_per_frame = []
        energy_per_frame = []
        energy_prob_per_frame = []

        iris= ""
        emotion = ""
        emotion_prob = 0
        energy = ""
        energy_prob = 0
        frameCounter = 0

        # load the video
        cap = cv2.VideoCapture('{}-input.mp4'.format(video_path))
        # get the fps of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("DEBUG: fps: {}".format(fps))
        #creating video writer to write the output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter('{}-temp.mp4'.format(video_path), fourcc, fps , frameSize = (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:

                print("video ended or failed to grab frame")
                break
            
            # get the iris position and facial emotion for the current frame
            iris, emotion, emotion_prob, energy, energy_prob, frame = facialModel.facialAnalysis(frame, emotion, emotion_prob, energy, energy_prob, frameCounter, fps, True)
            frameCounter += 1
            iris_pos_per_frame.append(iris)
            facial_emotion_per_frame.append(emotion)
            facial_emotion_prob_per_frame.append(emotion_prob)
            energy_per_frame.append(energy)
            energy_prob_per_frame.append(energy_prob)
            # write the frame to the output video
            output_video.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                 break
        cap.release()
        # release the video writer and close all windows
        cv2.destroyAllWindows()
        # calculate what is the most frequent energy
        energy = max(set(energy_per_frame), key=energy_per_frame.count)
        # delete the facial model object to free up memory
        del facialModel
        result_dict = {
            'iris_pos_per_frame': iris_pos_per_frame,
            'facial_emotion_per_frame': facial_emotion_per_frame,
            'facial_emotion_prob_per_frame': facial_emotion_prob_per_frame,
            'energy_per_frame': energy_per_frame,
            'energy_prob_per_frame': energy_prob_per_frame,
            'most_energy': energy
        }
        # put the result dictionary in the queue to be used by the main process
        queue.put(result_dict)
        return