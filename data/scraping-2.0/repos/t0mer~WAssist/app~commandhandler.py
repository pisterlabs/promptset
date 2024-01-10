import os
import time
import openai
import requests
import pytesseract
from  weatheril import WeatherIL
import numpy as np
import PyPDF2, pdfplumber
import file_dbaccess
import codecs
from pathlib import Path
from loguru import logger
from base64 import b64decode
from datetime import datetime, timedelta
from pydub import AudioSegment
from openai.embeddings_utils import get_embedding, cosine_similarity
import subprocess
from subprocess import Popen
from gcal import GoogleCal
try:
    from PIL import Image
except ImportError:
    import Image



#This code is based on the following repo:
#https://github.com/mangate/SelfGPT/blob/main/src/selfgpt.py

EMBEDDING_MODEL = 'text-embedding-ada-002'
COMPLETIONS_MODEL = "text-davinci-003"
QUESTION_COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 200,
    "model": COMPLETIONS_MODEL,
}

REGULAR_COMPLETIONS_API_PARAMS = {
    "temperature": 0.5,
    "max_tokens": 500,
    "model": COMPLETIONS_MODEL,
}

class CommandHandler:
    def __init__(self):
        self.openai = openai
        self.openai.api_key = os.getenv("OPENAI_KEY")
        self.db = file_dbaccess.LocalFileDbAccess("data/database.csv")
        self.db.ensureExists()
        # self.data_dir = Path.cwd() / "responses"
        self.image_dir = Path.cwd() / "images"
        self.audio_dir = Path.cwd() / "audio"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        
    def execute_command(self, msg:str):
        try:
            logger.debug("Got command: " + msg)
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            self.df = self.db.get()
            """
            Check the message and exexute relevant command
            """
            if msg.startswith("/h"):
                return("""Commands:\n\n/q [question] - Ask a question
                       \n/s [message] - Save your data 
                       \n/f [message] - Find related 
                       \n/d [message] - Generate
                       \n/w - Get weather forcast 
                       \n/e - Get events from calendar
                       \n/c - Show OpenAI estimated costs
                       \n/h - Show this help menu""")
                

            # Question answering
            elif msg.startswith("/q "):
                # Get the question
                question = str(msg.split("/q ")[1])
                # Construct the prompt
                prompt = self.construct_prompt(question, self.df, top_n=3)
                # Get the answer
                response = openai.Completion.create(prompt=prompt, **QUESTION_COMPLETIONS_API_PARAMS)
                return response["choices"][0]["text"]

            elif msg.startswith("/s "):
                data_to_save = msg.split("/s ")[1]
                # Save the massage to the database
                text_embedding = get_embedding(data_to_save, engine='text-embedding-ada-002')
                self.df = self.df.append({"time":dt_string,"message":data_to_save, "ada_search": text_embedding},ignore_index=True)
                self.db.save(self.df)
                return "Message saved successfully!"

            # Find related messages
            elif msg.startswith("/f "):
                query = str(msg.split("/f ")[1])
                most_similar = self.return_most_similiar(query, self.df, top_n=3)
                msg_reply = ''
                for i in range(len(most_similar)):
                    msg_reply += most_similar.iloc[i]['time'] + ': ' + most_similar.iloc[i]['message'] + '\n'
                return msg_reply

            elif msg.startswith("/d"):
                return self.generate_image(msg)

            elif msg.startswith("/c"):
                return self.get_usage()

            elif msg.startswith("/w"):
                return self.get_weather()

            elif msg.startswith("/e"):
                return GoogleCal().get_events()


            # Placeholder for other commands
            elif msg.startswith("/"):
                return("Sorry, I don't understand the command")


            # Get a regular completion
            else:
                # Just get a regular completion from the model
                COMPLETIONS_API_PARAMS = {
                    # We use temperature of 0.0 because it gives the most predictable, factual answer.
                    "temperature": 0.0,
                    "max_tokens": 200,
                    "model": COMPLETIONS_MODEL,
                }
                response = openai.Completion.create(prompt=msg, **REGULAR_COMPLETIONS_API_PARAMS)
                print (response)
                return response["choices"][0]["text"]
        except Exception as e:
            logger.error(str(e))
            return("aw snap something went wrong")
        
    def transcript(self, audio_file):
        try:
            audio_file = self.convertToWav(audio_file)
            data= open(audio_file, "rb")
            transcript = openai.Audio.transcribe("whisper-1", data)
            # os.remove(audio_file)
            return transcript["text"]
        except Exception as e:
            logger.error(str(e))
            return("aw snap something went wrong")

    def get_weather(self):
        try:
            forecast = ""
            weather = WeatherIL(1,"he").get_forecast() 
            for x in range(1, 5):
                forecast = forecast + f"*תחזית ארצית ליום {weather.days[x].day} ה {weather.days[x].date.strftime('%d/%m/%Y')}*\n"
                forecast = forecast + weather.days[x].description + "\n"
                forecast = forecast + f"טמפרטורה: {weather.days[x].maximum_temperature}°-{weather.days[x].minimum_temperature}°\n\n"
            return forecast
        except Exception as e:
            logger.error(e)
            return("aw snap something went wrong")

    def get_usage(self):
        try:
            ranges = {}
            usage = "*Your OpenAI Estimated costs are:*\n"
            headers = {"Authorization":"Bearer " + self.openai.api_key}
            ranges["*Today*"] = f"start_date={datetime.today().strftime('%Y-%m-%d')}&end_date={(datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')}" 
            ranges["*Yesterday*"] = f"start_date={(datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')}&end_date={datetime.today().strftime('%Y-%m-%d')}" 
            ranges["*Last_7_days*"] = f"start_date={(datetime.today() - timedelta(days=7)).strftime('%Y-%m-%d')}&end_date={datetime.today().strftime('%Y-%m-%d')}" 
            ranges["*Last_30_days*"] = f"start_date={(datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')}&end_date={datetime.today().strftime('%Y-%m-%d')}" 
            
            for key in ranges:
                url = f'https://api.openai.com/dashboard/billing/usage?{ranges[key]}'
                result = requests.get(url, headers=headers)
                costs = float(result.json()["total_usage"])/100
                usage = usage + key.replace("_", " ") + ": " +  str(round(costs,3))+" $\n"
            return usage
        except Exception as e:
            logger.error(str(e))
            return("aw snap something went wrong") 

    def generate_image(self, prompt):
        try:
            response = self.openai.Image.create(
            prompt=prompt,
            n=1,
            size="1024x1024",
            response_format="b64_json",
            )


            for index, image_dict in enumerate(response["data"]):
                image_data = b64decode(image_dict["b64_json"])
                image_file = self.image_dir / f"{response['created']}.png"
                with open(image_file, mode="wb") as png:
                    png.write(image_data)
                return str(image_file)
        except Exception as e:
            logger.error(str(e))
            return str(e)

    def construct_prompt(self,question, df, top_n=3):
        try:
            # Get the context
            context = self.generate_context(question, df, top_n)
            header =  header = """Answer the question in details, based only on the provided context and nothing else, and if the answer is not contained within the text below, say "w.", do not invent or deduce!\n\nContext:\n"""
            return header + "".join(context) + "Q: " + question + "\n A:"
        except Exception as e:
            logger.error(str(e))
            return("aw snap something went wrong")

    def generate_context(self,question, df, top_n=3):
        try:
            most_similiar = self.return_most_similiar(question, df, top_n)
            # Get the top 3 most similar messages
            top_messages = most_similiar["message"].values
            # Concatenate the top 3 messages into a single string
            context = '\n '.join(top_messages)
            return context
        except Exception as e:
            logger.error(str(e))

    def return_most_similiar(self,question, df, top_n=3):
        try:
            # Get the embedding for the question
            
            question_embedding = get_embedding(question, engine='text-embedding-ada-002')
            # Get the embedding for the messages in the database
            df["ada_search"] = df["ada_search"].apply(eval).apply(np.array)
            # Get the similarity between the question and the messages in the database
            df['similarity'] = df.ada_search.apply(lambda x: cosine_similarity(x, question_embedding))
            # Get the index of the top 3 most similar message
            most_similiar = df.sort_values('similarity', ascending=False).head(top_n)
            return most_similiar
        except Exception as e:
            logger.error(str(e))


    


    #Get audio file extention
    def getExtention(self, audio_file):
        logger.info("Getting audio file extention")
        filename, file_extension = os.path.splitext(audio_file)
        return filename, file_extension

    # Convert to wav if needed
    def convertToWav(self, audio_file):
        logger.info("Input File:" +  audio_file)
        filename, file_extension = self.getExtention(audio_file)
        output_file = filename + ".wav"
        logger.info("Output File:" +  output_file)
        if file_extension == ".mp3":
            logger.info("Converting from mp3 to WAV")
            sound = AudioSegment.from_mp3(audio_file)
            sound.export(output_file, format="wav")
        if file_extension == ".ogg":
            logger.info("Converting from ogg to WAV")
            sound = AudioSegment.from_ogg(audio_file)
            sound.export(output_file, format="wav")
        if file_extension == ".mp4":
            logger.info("Converting from mp4 to WAV")
            sound = AudioSegment.from_file(audio_file, "mp4")
            sound.export(output_file, format="wav")
        if file_extension == ".wma":
            logger.info("Converting from wma to WAV")
            sound = AudioSegment.from_file(audio_file, "wma")
            sound.export(output_file, format="wav")
        if file_extension == ".aac":
            logger.info("Converting from aac to WAV")
            sound = AudioSegment.from_file(audio_file, "aac")
            sound.export(output_file, format="wav")
        os.remove(audio_file)
        return output_file


    def extract_text_from_pdf(self,file,documents_dir):
        file = str(file)
        txt_file = documents_dir/os.path.basename(file).split('/')[-1].lower().replace('pdf','txt')
        if os.path.isfile(file) and file.endswith(".pdf"):
            # Open the PDF file in read-binary mode
            text = ""
            with open(file, "rb") as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                
                # Extract text from each page of the PDF
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
                
            with open(txt_file, "w") as txt_file:
                txt_file.write(text)
            
            
            os.remove(file)



    # def extract_text_from_pdf(self,file,learn_dir):
    #     try:
    #         return True
    #         file = self.convert_to_tiff(str(file))
    #         txt_file = learn_dir/os.path.basename(file).split('/')[-1].lower().replace('tiff','txt')
    #         if os.path.isfile(file) and file.endswith(".tiff"):
    #             text= pytesseract.image_to_string(Image.open(file), lang="heb+eng")
    #             with open(txt_file, "w") as txt_file:
    #                 txt_file.write(text)
    #             os.remove(file)
                
    #     except Exception as e:
    #         logger.error(str(e))            




    # def convert_to_tiff(self,image_file):
    #     logger.info("Converting pdf to tiff")
    #     converted_file_name = image_file.replace('pdf','tiff')
    #     p = subprocess.Popen('convert -density 300 '+ image_file +' -background white -alpha Off '+ converted_file_name , stderr=subprocess.STDOUT, shell=True)
    #     p_status = p.wait()
    #     time.sleep(5)
    #     if os.path.exists(image_file):
    #         os.remove(image_file)
    #     return converted_file_name