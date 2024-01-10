from filecmp import clear_cache
from re import T
import pyaudio
import streamlit as st
import websockets
import asyncio
import base64
import json
import openai
import pyttsx3
import time 
import os
from api_keys import assemblyAI_key, openaI_key
openai.api_key = openaI_key


from google.cloud import aiplatform


def create_custom_job_sample(
    caption: str,
    project: str,
    display_name: str,
    container_image_uri: str,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.JobServiceClient(client_options=client_options)
    custom_job = {
        "display_name": display_name,
        "job_spec": {
            "worker_pool_specs": [
                {
                    "machine_spec": {
                        "machine_type": "n1-standard-4",
                        "accelerator_type": aiplatform.gapic.AcceleratorType.NVIDIA_TESLA_K80,
                        "accelerator_count": 1,
                    },
                    "replica_count": 1,
                    "python_package_spec": {
                        "executor_image_uri": container_image_uri,
                        "package_uris" : ["gs://image_gen_jobs/image_generation-0.2.tar.gz"],
                        "python_module": "trainer.task",
                        "args":["--caption={}".format(caption)]
                        },
                }
            ]
        },
    }
    parent = f"projects/{project}/locations/{location}"
    response = client.create_custom_job(parent=parent, custom_job=custom_job)
    print("response:", response)
      

class app:
   def __init__(self):

      self.FRAMES_PER_BUFFER = 3200
      self.FORMAT = pyaudio.paInt16
      self.CHANNELS = 1
      self.RATE = 16000
      self.p = pyaudio.PyAudio()

      # the AssemblyAI endpoint we're going to hit
      self.URL = "wss://api.assemblyai.com/v2/realtime/ws?sample_rate=16000"
      
      self.bot_text, self.user_text = [], []
      self.pipeline()



   def pipeline(self):
         self.initialize_tool()
         self.buttons()
         self.load_past()
         asyncio.run(self.send_receive())


   def initialize_tool(self):
      
      # init streamlit app
      st.set_page_config(
         page_title="Interactive AI", page_icon="🤖" 
      )

      st.markdown('<h1 style="color: white">SMART ASSISTANT TOOL</h1>', unsafe_allow_html=True)


      # init recording 
      self.stream = self.p.open(
         format=self.FORMAT,
         channels=self.CHANNELS,
         rate=self.RATE,
         input=True,
         frames_per_buffer=self.FRAMES_PER_BUFFER
      )

      # init session state 
      if "init" not in st.session_state:
         st.session_state["init"] = False
            

   def toggle_on(self):
      st.session_state["init"] = True

   def toggle_off(self):
      st.session_state["init"] = False

   def clear_chat(self):
      if os.path.exists("chat1.txt"):
         os.remove("chat1.txt")

      if os.path.exists("chat2.txt"):
         os.remove("chat2.txt")

      with open('chat1.txt', 'x') as f:
         f.write("")

      with open('chat2.txt', 'x') as f:
         f.write("")

   

   def buttons(self):
      col1, col2 = st.columns((1,1))
      with col1:
         st.markdown("## ")
         st.markdown("## ")
         st.button("Record", on_click = self.toggle_on)
         st.button("Clear Chat", on_click = self.clear_chat)

      # with col2:
         # st.image("oldman1.png", width=300)


      self.speaker1, space, self.speaker2 = st.columns((1, 0.2, 1))
      with self.speaker1:
         st.markdown('<h2 style="color: white">USER</h2>', unsafe_allow_html=True) 

      with self.speaker2:
         st.markdown('<h2 style="color: pink; text-align:right">BOT</h2>', unsafe_allow_html=True)




   def load_past(self):
      # LOAD PAST MESSAGES

      with open ("chat1.txt", "r") as myfile:
         user_text = myfile.read().splitlines()

      with open ("chat2.txt", "r") as myfile:
         bot_text = myfile.read().splitlines()

      for i, j in zip(user_text, bot_text):
         with self.speaker1:
               st.markdown("## ")
               st.markdown('<p style="color: white;  font-size:25px">{}</p>'.format(i),
                                             unsafe_allow_html=True)
               st.markdown("## ")
         with self.speaker2:
            st.markdown("## ")
            st.markdown("## ")
            st.markdown('<p style="color: pink; text-align:right; font-size:25px">{}</p>'.format(j),  unsafe_allow_html=True)
            st.markdown("## ")



   def generate_art(self, text):
      t = text.split("Generate")[-1]

      with self.speaker2:

            st.markdown("## ")
            st.markdown("## ")
            
            # launch job to vertex
            create_custom_job_sample(
                 caption = "t",
                 project = "drone-swarm",
                 container_image_uri = "europe-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-10:latest",
                 display_name = "art_gen_script")
            
            # add delay for job to run
            time.sleep(300)

            #load image from bucket
            

            from google.cloud import storage
          
            client = storage.Client()

            bucket = client.get_bucket('image_gen')

            blob = bucket.get_blob('image_gen/{}.png'.format(t))
            blob.download_to_filename('{}.png'.format(t))

            #show image
            from PIL import Image
            image = Image.open('{}.png'.format(t))

            st.image(image, caption=t)


   async def send_receive(self):
      print(f'Connecting websocket to url ${self.URL}')
      async with websockets.connect(
         self.URL,
         extra_headers=(("Authorization", assemblyAI_key),),
         ping_interval=5,
         ping_timeout=20
      ) as _ws:
         r = await asyncio.sleep(0.1)
         print("Receiving SessionBegins ...")
         session_begins = await _ws.recv()

         async def send():
               while st.session_state["init"] == True:
                  try:
                     data = self.stream.read(self.FRAMES_PER_BUFFER, exception_on_overflow = False)
                     data = base64.b64encode(data).decode("utf-8")
                     json_data = json.dumps({"audio_data":str(data)})
                     r = await _ws.send(json_data)
                  except websockets.exceptions.ConnectionClosedError as e:
                     print(e)
                     assert e.code == 4008
                     break
                  except Exception as e:
                     assert False, "Not a websocket 4008 error"
                  r = await asyncio.sleep(0.01)
               
               if st.session_state["init"] == False:
                  closeAPI = json.dumps({"terminate_session": True})
                  r = await _ws.send(closeAPI)

               return True

         async def receive():

               while st.session_state["init"] == True:

                  try:
                     result_str = await _ws.recv()
                     if (json.loads(result_str)["message_type"] == "FinalTranscript") and (json.loads(result_str)['text'] != ""):
                        

                        # user_text.append(json.loads(result_str)['text'])
                        with open('chat1.txt', 'a') as f:
                              f.write(json.loads(result_str)['text'] + '\n')

                        with self.speaker1:
                           st.markdown("## ")
                           text = json.loads(result_str)['text']
                           st.markdown('<p style="color: white;  font-size:25px">{}</p>'.format(text),
                                                         unsafe_allow_html=True)
                           st.markdown("## ")
                        if "Generate" in text:
                           self.generate_art(text)
                        else:
                           promt = json.loads(result_str)["text"]
                           response = openai.Completion.create(
                              engine = "text-davinci-002",
                              prompt = promt,
                              n=5,
                              temperature=0.7,
                              max_tokens=80,
                              top_p=1,
                              frequency_penalty=0,
                              presence_penalty=0
                           )
                           print(response)

                           response_test = response.choices[0].text
                           response_test = response_test.replace("\n", "")
                           with self.speaker2:
                              st.markdown("## ")
                              st.markdown("## ")
                              st.markdown('<p style="color: pink; text-align:right; font-size:25px">{}</p>'.format(response_test),  unsafe_allow_html=True)
                              st.markdown("## ")

                           with open('chat2.txt', 'a') as f:
                                 f.write(response_test + '\n')

                           pyttsx3.speak(response_test.replace("\u03c0", "pi"))
                        self.toggle_off()

                  except Exception as e:
                     st.write("ERROR", e)
                     assert False

         send_result, receive_result = await asyncio.gather(send(), receive())



if __name__ == '__main__':
    app()


