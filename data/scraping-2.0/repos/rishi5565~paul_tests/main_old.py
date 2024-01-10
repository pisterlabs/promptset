# import dependencies
from flask import Flask, request, jsonify
from googletrans import Translator, LANGUAGES
from transformers import pipeline
import openai
from openai.error import AuthenticationError
import base64
import os
import pytesseract
import cv2
from PIL import Image
import io
from io import BytesIO
from gtts import gTTS
from datetime import datetime
import re
import random
import string
import requests
import subprocess
import uuid
import magic
from pydub import AudioSegment
import io
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from time import sleep
import boto3
import hashlib
import time
import tiktoken
from gensim.utils import simple_preprocess

import logging
from logging.handlers import TimedRotatingFileHandler

logFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
rootLogger = logging.getLogger()

fileHandler = TimedRotatingFileHandler('AppLog.log', when="midnight", interval=1, backupCount=7)
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

logger = logging.getLogger(__name__)

rootLogger.setLevel(logging.INFO)


# load models
# audio_to_text_model = whisper.load_model("small")
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
# logger.info("Summarization model loaded.")


# define AWS params
os.environ['AWS_ACCESS_KEY_ID'] = ""
os.environ['AWS_SECRET_ACCESS_KEY'] = ""
os.environ['AWS_REGION'] = "us-east-1"

# define OpenAI params
model_engine = "text-davinci-002"
max_window_length = 1800


# define functions


def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def generate_reference_number(length):
    characters = string.ascii_letters + string.digits
    reference_number = ''.join(random.choice(characters) for _ in range(length))
    return reference_number.upper()


def detect_language(text):
    try:
        translator = Translator()
        detected = translator.detect(text)
        lang_code = detected.lang
        return lang_code
    except Exception as e:
        logger.error(f"Error in detect_language: \n{e}")
        raise e



def detect_file_type(base64_string):
    # Decode base64 string into bytes
    file_bytes = base64.b64decode(base64_string, validate=True)

    # Determine the file type using magic number
    file_type = magic.from_buffer(file_bytes, mime=True)

    return file_type


def convert_to_audio_mpeg(base64_string):
    # Decode base64 string into bytes
    audio_bytes = base64.b64decode(base64_string, validate=True)

    # Create an AudioSegment from the bytes
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))

    # Convert to audio/mpeg format
    converted_audio = audio.export(format="mp3")

    # Encode the converted audio data back to base64
    converted_base64 = base64.b64encode(converted_audio.read()).decode("utf-8")

    return converted_base64
  

def audio_to_trasncript(api_key, b64_audio, audio_extension, need_audio=False):
    try:

        b64_type = detect_file_type(b64_audio)

        if b64_type != "audio/mpeg":
            b64_audio = convert_to_audio_mpeg(b64_audio)

        openai.api_key = api_key
        decoded_audio = base64.b64decode(b64_audio)
        temp_file_name = f'temp_{uuid.uuid4()}.mp3'

        with open(temp_file_name, 'wb') as temp_audio_file:
            temp_audio_file.write(decoded_audio)
        
        transcript = (openai.Audio.translate("whisper-1", open(temp_file_name, "rb"))).to_dict()['text']
        
        # Delete the temporary file after usage
        os.remove(temp_file_name)

        translator = Translator()
        output_language = detect_language(transcript)

        if output_language.lower() not in ['en']:
            try:
                translated_transcript = translator.translate(transcript, dest=output_language).text
            except:
                pass
        else:
            translated_transcript = transcript

        if need_audio == True:
            try:
                max_retries = 5
                while max_retries <= 5:
                    try:
                        myobj = gTTS(text=translated_transcript, lang=output_language, slow=False)
                        temp_audio_file_name = f'temp_{uuid.uuid4()}.mp3'
                        myobj.save(temp_audio_file_name)
                        break
                    except:
                        max_retries += 1
            except ValueError:
                max_retries = 5
                while max_retries <= 5:
                    try:
                        myobj = gTTS(text=translated_transcript, lang='en', slow=False)
                        temp_audio_file_name = f'temp_{uuid.uuid4()}.mp3'
                        myobj.save(temp_audio_file_name)
                        break
                    except:
                        max_retries += 1

                logger.info(f"gTTS could not convert to {output_language}. Converted to English instead.")

            with open(temp_audio_file_name, 'rb') as audio_file:
                audio_bytes = audio_file.read()

            os.remove(temp_audio_file_name)

            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

            return audio_base64, translated_transcript
        else:
            return translated_transcript

    except Exception as e:
        logger.error(f"Error in audio_to_trasncript: \n{e}\n")
        logger.info(f"Input details:\nAPI Key: {api_key}\nB64 Audio: {b64_audio}\nNeed Audio: {need_audio}")
        raise e


def summarize_langchain(api_key, ARTICLE, need_audio=False):
  try:

    prompt = """You will be provided with an article or a piece of text. You will have to summarize it in a few sentences whichever you find to be of appropriate length. Also you will have to ensure the summary and the article are of the same language. Your output should be in the following format: \n\n\
        Summary: {the summary goes here}
        
        Reply with "READ" if you understand.
        """
    
    user_input = f"Article: {ARTICLE}"

    llm = ChatOpenAI(
    model_name = "gpt-3.5-turbo-16k",
    temperature = 0.8,
    openai_api_key=api_key,
    request_timeout=60,
    )

    history = ChatMessageHistory()
    history.add_user_message(prompt)
    history.add_ai_message('READ')

    memory = ConversationBufferMemory(chat_memory=history)

    conversation = ConversationChain(
    llm=llm,
    memory=memory)

    translated_summary = conversation.predict(input=user_input)

    output_language = detect_language(translated_summary)


    if need_audio == True:
        try:
            max_retries = 5
            while max_retries <= 5:
                try:
                    myobj = gTTS(text=translated_summary, lang=output_language, slow=False)
                    myobj.save('sm_temp.mp3')
                    break
                except:
                    max_retries += 1

        except ValueError:
            max_retries = 5
            while max_retries <= 5:
                try:
                    myobj = gTTS(text=translated_summary, lang='en', slow=False)
                    myobj.save('sm_temp.mp3')
                    break
                except:
                    max_retries += 1

                logger.info(f"gTTS could not convert to {output_language}. Converted to English instead.")
        
        # Open the audio file
        with open('sm_temp.mp3', 'rb') as audio_file:
            # Read the audio file contents
            audio_bytes = audio_file.read()

        # Convert the audio to base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        return audio_base64, translated_summary
    else:
        return translated_summary
    
  except Exception as e:
    logger.error(f"Error in summarize: \n{e}\n")
    logger.info(f"Input details:\nArticle: {ARTICLE}\nNeed Audio: {need_audio}")





def answer_question(qna_dict, api_key, context=None):
    try:
        openai.api_key = api_key
        messages_list = [{"role": "system", "content": "You are a helpful assistant."}]
        for key in qna_dict:
            for i, (question, answer) in enumerate(qna_dict[key].items()):
                if i == 0 and context:
                    question = f"{context} \n\n{question}"
                messages_list.append({"role": "user", "content": question})
                if answer != "unanswered":
                    messages_list.append({"role": "assistant", "content": answer}) 
        messages = messages_list
        chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages
            )
        answer = chat.choices[0].message.content
        output_language = detect_language(answer)
        if output_language.lower() not in['en', 'english']:
            translator = Translator()
            answer = translator.translate(answer, dest=output_language).text
        return answer
    except Exception as e:
        logger.error(f"Error in answer_question: \n{e}\n")
        logger.info(f"Input details:\nQnA Dict: {qna_dict}\nAPI Key: {api_key}\nContext: {context}")
        raise e


def is_base64(s):
    try:
        if re.match(r'^[A-Za-z0-9+/]*={0,2}$', s):
            base64.b64decode(s)
            return True
        else:
            return False
    except Exception:
        logger.info("Returning False from Exception (is_base64)")
        return False
    


def answer_question_v2_langchain(qna_dict, api_key, context=None, need_audio=False):
    try:
        history = ChatMessageHistory()
        history.add_user_message("""You are ChatGPT, an assistant. You never apologize for things such as not knowing the answer to a question. You also never apologize for repetition. In general, you are very confident in your abilities and you never apologize for anything. Reply with READ if you understand.""")
        history.add_ai_message("READ")

        for key in qna_dict:
            for i, (question, answer) in enumerate(qna_dict[key].items()):
                # Check if the question is in base64 format
                if is_base64(question):
                    # Decode the base64 audio and convert it to text
                    question_transcript = audio_to_trasncript(api_key, question, 'mp3')
                    question = question_transcript

                if i == 0 and context:
                    question = f"{context} \n\n{question}"

                # Add user message
                history.add_user_message(question)

                if answer != "unanswered":
                    # Add assistant message
                    history.add_ai_message(answer)

        total_tokens = sum([num_tokens_from_string(i.content, 'gpt-3.5-turbo') for i in history.messages])

        while total_tokens > 15000:
            del history.messages[0]
            total_tokens = sum([num_tokens_from_string(i.content, 'gpt-3.5-turbo') for i in history.messages])

        memory = ConversationBufferMemory(chat_memory=history)

        llm = ChatOpenAI(
        model_name = "gpt-3.5-turbo-16k",
        temperature = 0.8,
        openai_api_key=api_key,
        request_timeout=120,
        )

        conversation = ConversationChain(
        llm=llm,
        memory=memory)

        answer = conversation.predict(input=question)
        answer_sentences = answer.split(".")
        filtered_answer_sentences = []
        for sent in answer_sentences:
            if ("apologize" in " ".join(simple_preprocess(sent))) or ("apologies" in " ".join(simple_preprocess(sent))):
                if ("repetition" in " ".join(simple_preprocess(sent))) or ("repeat" in " ".join(simple_preprocess(sent))):
                        continue
            else:
                filtered_answer_sentences.append(sent)
        answer = ".".join(filtered_answer_sentences)

        output_language = detect_language(answer)

        if output_language.lower() not in ['en', 'english']:
            try:
                translator = Translator()
                answer = translator.translate(answer, dest=output_language).text
            except:
                answer = answer

        if need_audio:
            try:
                max_retries = 5
                while max_retries <= 5:
                    try:
                        myobj = gTTS(text=answer, lang=output_language, slow=False)
                        myobj.save('qna_temp.mp3')
                        break
                    except:
                        max_retries += 1
            except ValueError:
                max_retries = 5
                while max_retries <= 5:
                    try:
                        myobj = gTTS(text=answer, lang='en', slow=False)
                        myobj.save('qna_temp.mp3')
                        break
                    except:
                        max_retries += 1
                        logger.info(f"gTTS could not convert to {output_language}. Converted to English instead.")

            # Open the audio file
            with open('qna_temp.mp3', 'rb') as audio_file:
                # Read the audio file contents
                audio_bytes = audio_file.read()

            # Convert the audio to base64
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

            return audio_base64, answer
        else:
            return answer

    except Exception as e:
        logger.error(f"Error in answer_question: \n{e}\n")
        logger.info(f"Input details:\nQnA Dict: {qna_dict}\nAPI Key: {api_key}\nContext: {context}\nNeed Audio: {need_audio}")
        raise e


def answer_question_v2(qna_dict, api_key, context=None, need_audio=False):
    try:
        openai.api_key = api_key
        messages_list = [{"role": "system", "content": "You are a helpful assistant."}]
        for key in qna_dict:
            for i, (question, answer) in enumerate(qna_dict[key].items()):
                # Check if the question is in base64 format
                if is_base64(question):
                    # Decode the base64 audio and convert it to text
                    question_transcript = audio_to_trasncript(api_key, question, 'mp3')
                    question = question_transcript

                if i == 0 and context:
                    question = f"{context} \n\n{question}"

                messages_list.append({"role": "user", "content": question})
                if answer != "unanswered":
                    messages_list.append({"role": "assistant", "content": answer})

        messages = messages_list
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
        answer = chat.choices[0].message.content
        output_language = detect_language(answer)

        if output_language.lower() not in ['en', 'english']:
            try:
                translator = Translator()
                answer = translator.translate(answer, dest=output_language).text
            except:
                pass
            
        if need_audio == True:
            try:
                max_retries = 5
                while max_retries <= 5:
                    try:
                        myobj = gTTS(text=answer, lang=output_language, slow=False)
                        myobj.save('qna_temp.mp3')
                        break
                    except:
                        max_retries += 1
            except ValueError:
                max_retries = 5
                while max_retries <= 5:
                    try:
                        myobj = gTTS(text=answer, lang='en', slow=False)
                        myobj.save('qna_temp.mp3')
                        break
                    except:
                        max_retries += 1
                        logger.info(f"gTTS could not convert to {output_language}. Converted to English instead.")
        
            # Open the audio file
            with open('qna_temp.mp3', 'rb') as audio_file:
                # Read the audio file contents
                audio_bytes = audio_file.read()

            # Convert the audio to base64
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

            return audio_base64, answer
        else:
            return answer
    
    except Exception as e:
        logger.error(f"Error in answer_question: \n{e}\n")
        logger.info(f"Input details:\nQnA Dict: {qna_dict}\nAPI Key: {api_key}\nContext: {context}\nNeed Audio: {need_audio}")
        raise e




def img_to_text(b64_img):
    try:
        # Decode the b64 string into bytes
        image_bytes = base64.b64decode(b64_img)
        # Open the image using PIL
        image = Image.open(BytesIO(image_bytes))
        # Save the image as a file
        image.save("temp_img.png")
        image = cv2.imread('temp_img.png')
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply a bilateral filter to smooth out the image while preserving edges
        blur = cv2.bilateralFilter(gray, 11, 17, 17)
        # Extract text from the image using pytesseract
        text = pytesseract.image_to_string(blur)
        # Return the extracted text
        return text
    except Exception as e:
        logger.error(f"Error in img_to_text: \n{e}\n")
        logger.info(f"Input details:\nB64 Image: {b64_img}")
        raise e
    



def img_to_text_external(b64_img, api_key="K89645445888957", language='eng'):
    try:
        payload = {
            'isOverlayRequired': False,
            'apikey': api_key,
            'language': language
        }

        file_data = base64.b64decode(b64_img)
        files = {'filename': ('image.' + "png", file_data, 'image/' + "png")}

        r = requests.post('https://api.ocr.space/parse/image',
                        files=files,
                        data=payload
                        )

        return r.json()["ParsedResults"][0]["ParsedText"]
    except Exception as e:
        logger.error(f"Error in img_to_text_external: \n{e}\n")
        logger.info(f"Input details:\nB64 Image: {b64_img}\nAPI Key: {api_key}\nLanguage: {language}")
        raise e




def img_to_text_aws_textract(b64_img, bucket_name="img-hold-server"):
    
    s3 = boto3.client('s3')

    sha256_hash = hashlib.sha256()
    sha256_hash.update(b64_img.encode('utf-8'))
    hash_value = sha256_hash.hexdigest()

    local_file_path = f"{hash_value}.png"

    image_bytes = base64.b64decode(b64_img)
    image = Image.open(BytesIO(image_bytes))
    image.save(local_file_path)


    with open(local_file_path, 'rb') as f:
        file_content = f.read()
        file_hash = hashlib.sha256(file_content).hexdigest()

    s3_key = f'images/{file_hash}'

    s3.upload_file(local_file_path, bucket_name, s3_key)

    textract_client = boto3.client('textract', region_name = 'us-east-1')

    response = textract_client.start_document_text_detection(
        DocumentLocation={'S3Object': {'Bucket': 'img-hold-server', 'Name': f'{s3_key}'}}
    )

    job_id = response['JobId']

    for i in range(12):
        try:
            response = textract_client.get_document_text_detection(JobId=job_id)
            extracted_text_list = []
            blocks = response['Blocks']
            if response:
                break
        except Exception as e:
            time.sleep(5)

    for block in blocks:
        if block['BlockType'] == 'LINE':
            extracted_text_list.append(block['Text'])

    extracted_text = "\n".join(extracted_text_list)

    os.remove(local_file_path)
    s3.delete_object(Bucket=bucket_name, Key=s3_key)

    return extracted_text



def sentiment_analysis(passage, api_key):
    try:
        openai.api_key = api_key
        translator = Translator()
        passage = translator.translate(passage, dest='en').text
        prompt = f"""Detect the sentiment in the following passage. 
                    The sentiment could be Happiness, Sadness, Anger, Love, Fear, Excitement, Enthusiasm, Frustration, 
                    Confusion, Surprise, Disgust, Guilt, Shame, Jealousy, Envy, Gratitude, Sympathy, Empathy, Compassion, 
                    Contentment, Satisfaction, Disappointment, Regret, Hope, Optimism, Pessimism, Nostalgia, Amusement, Boredom.\n 
                    I want only one-word answer as response.\n{passage}"""

        messages = [{"role": "user", "content": prompt}]
        # Send the messages to OpenAI's GPT-3 model
        chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k", messages=messages
            )
        reply = chat.choices[0].message.content
        output_language = detect_language(reply)
        if output_language.lower() not in['en', 'english']:
            try:
                translator = Translator()
                reply = translator.translate(reply, dest=output_language).text
            except:
                pass
        # Return the response
        return reply
    except Exception as e:
        logger.error(f"Error in sentiment_analysis: \n{e}\n")
        logger.info(f"Input details:\nPassage: {passage}\nAPI Key: {api_key}")
        raise e

def generate_audio_b64(text, filename='output_audio.mp3'):
    try:
        lang = detect_language(text)
        try:
            myobj = gTTS(text=text, lang=lang, slow=False)
            myobj.save(filename)
        except ValueError:
            myobj = gTTS(text=text, lang='en', slow=False)
            myobj.save(filename)
            logger.info(f"gTTS could not convert to {lang}. Converted to English instead.")
        
        # Open the audio file
        with open('output_audio.mp3', 'rb') as audio_file:
            # Read the audio file contents
            audio_bytes = audio_file.read()

        # Convert the audio to base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        return audio_base64
    except Exception as e:
        logger.error(f"Error in generate_audio_b64: \n{e}\n")
        logger.info(f"Input details:\nText: {text}\nFilename: {filename}")
        raise e
    

def process_file(api_key, file):

    os.environ["OPENAI_API_KEY"] = api_key
    
    # Load the PDF file using PyPDFLoader
    loader = PyPDFLoader(file.name) 
    documents = loader.load()
    
    # Initialize OpenAIEmbeddings for text embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create a ConversationalRetrievalChain with ChatOpenAI language model
    # and PDF search retriever
    pdfsearch = Chroma.from_documents(documents, embeddings,)

    chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.3, model_name = "gpt-3.5-turbo-16k"), 
                                                  retriever=
                                                  pdfsearch.as_retriever(search_kwargs={"k": 1}),
                                                  return_source_documents=True)
    return chain



def generate_response(api_key, file, query):
    # global chat_history
    chat_history = []
    
    chain = process_file(api_key, file)
    
    # Generate a response using the conversation chain
    result = chain({"question": query, 'chat_history':chat_history}, return_only_outputs=True)
    
    # Update the chat history with the query and its corresponding answer
    # chat_history += [(query, result["answer"])]
    
    # Retrieve the page number from the source document
    N = list(result['source_documents'][0])[1][1]['page']

    return {"answer": result["answer"], "ref_pg_no": N}


def save_pdf_file(b64_pdf):
    pdf_file = base64.b64decode(b64_pdf)
    hash_name = hashlib.md5(pdf_file).hexdigest()
    with open(f'{hash_name}.pdf', 'wb') as f:
        f.write(pdf_file)
    return {'pdf_id': hash_name, 'status': 'success'}


def chat_pdf(api_key, hash_name, ques):
    file = open(f'{hash_name}.pdf', 'rb')
    response = generate_response(api_key, file, ques)
    return response




logger.info("All functions loaded.")


get_text = "The API is working. Please send POST request."

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

@app.route('/a2t', methods=['POST', 'GET'])
def a2t():
    try:

        if not os.path.exists("a2t_counter.txt"):
            with open("a2t_counter.txt", "w") as f:
                f.write("0")

        with open("a2t_counter.txt", "r") as f:
            counter = int(f.read())

        with open("a2t_counter.txt", "w") as f:
            f.write(str(counter + 1))

        if (request.method == "POST"):
            b64_audio = request.get_json().get('b64_audio')
            audio_extension = request.get_json().get('audio_extension')
            need_audio = request.get_json().get('need_audio')
            api_key = request.get_json().get('api_key')
            if int(need_audio) == 1:
                audio_f, transcript = audio_to_trasncript(api_key, b64_audio, audio_extension, True)
                data_dict = {'transcript': transcript, 'b64_of_transcript': audio_f}
                return jsonify(data_dict) 
            else:
                transcript = audio_to_trasncript(api_key, b64_audio, audio_extension)
                data_dict = {'transcript': transcript}
                return jsonify(data_dict)
        elif (request.method == "GET"):
            return get_text
    except Exception as e:
        reference_number = generate_reference_number(8)
        logger.error(f"Error in /a2t route [Ref: {reference_number}]: \n{e}\n")
        return jsonify(f'Error occurred. Please check server log with Ref No. {reference_number}\nError Summary: {e}')


@app.route('/summarize', methods=['POST', 'GET'])
def summ():
    try:

        if not os.path.exists("summarize_counter.txt"):
            with open("summarize_counter.txt", "w") as f:
                f.write("0")

        with open("summarize_counter.txt", "r") as f:
            counter = int(f.read())

        with open("summarize_counter.txt", "w") as f:
            f.write(str(counter + 1))


        if (request.method == "POST"):
            article = request.get_json().get('article')
            need_audio = request.get_json().get('need_audio')
            api_key = request.get_json().get('api_key')
            if int(need_audio) == 1:
                # audio_f, summary = summarize(article, True)
                audio_f, summary = summarize_langchain(api_key, article, True)
                data_dict = {'summary': summary, 'b64_of_summary': audio_f}
                return jsonify(data_dict)
            else:
                # summary = summarize(article)
                summary = summarize_langchain(api_key, article)
                data_dict = {'summary': summary}
                return jsonify(data_dict)
        elif (request.method == "GET"):
            return get_text
    except Exception as e:
        reference_number = generate_reference_number(8)
        logger.error(f"Error in /summarize route [Ref: {reference_number}]: \n{e}\n")
        return jsonify(f'Error occurred. Please check server log with Ref No. {reference_number}\nError Summary: {e}')


@app.route('/q2a', methods=['POST', 'GET'])
def q2a():
    try:

        if not os.path.exists("q2a_counter.txt"):
            with open("q2a_counter.txt", "w") as f:
                f.write("0")

        with open("q2a_counter.txt", "r") as f:
            counter = int(f.read())

        with open("q2a_counter.txt", "w") as f:
            f.write(str(counter + 1))

        if (request.method == "POST"):
            qna_dict = request.get_json().get('qna_dict')
            # Convert all keys to integers
            qna_dict = {int(k): v for k, v in qna_dict.items()}
            context = request.get_json().get('context')
            api_key = request.get_json().get('api_key')
            try:
                if int(context) == 0:
                    answer = answer_question(qna_dict, api_key)
                else:
                    answer = answer_question(qna_dict, api_key, context)
                return jsonify(answer)
            except AuthenticationError:
                return jsonify('OpenAI API Key failed to authenticate.')
        elif (request.method == "GET"):
            return get_text
    except Exception as e:
        reference_number = generate_reference_number(8)
        logger.error(f"Error in /q2a route [Ref: {reference_number}]: \n{e}\n")
        return jsonify(f'Error occurred. Please check server log with Ref No. {reference_number}\nError Summary: {e}')
    

@app.route('/q2a_test', methods=['POST', 'GET'])
def q2a_test():
    try:

        if not os.path.exists("q2a_test_counter.txt"):
            with open("q2a_test_counter.txt", "w") as f:
                f.write("0")

        with open("q2a_test_counter.txt", "r") as f:
            counter = int(f.read())

        with open("q2a_test_counter.txt", "w") as f:
            f.write(str(counter + 1))

        if (request.method == "POST"):
            qna_dict = request.get_json().get('qna_dict')
            # Convert all keys to integers
            qna_dict = {int(k): v for k, v in qna_dict.items()}
            context = request.get_json().get('context')
            api_key = request.get_json().get('api_key')
            need_audio = request.get_json().get('need_audio')
            try:
                int(context)
                no_context = True
            except:
                no_context = False

            try:
                if no_context and (int(need_audio) == 1):
                    # audio_b64, answer = answer_question_v2(qna_dict, api_key, context=None, need_audio=True)
                    audio_b64, answer = answer_question_v2_langchain(qna_dict, api_key, context=None, need_audio=True)
                    data_dict = {'answer': answer, 'b64_of_answer': audio_b64}
                    return jsonify(data_dict)
                
                elif no_context and (int(need_audio) == 0):
                    # answer = answer_question_v2(qna_dict, api_key, context=None, need_audio=False)
                    answer = answer_question_v2_langchain(qna_dict, api_key, context=None, need_audio=False)
                    data_dict = {'answer': answer}
                    return jsonify(data_dict)
                
                elif (no_context == False) and (int(need_audio) == 1):
                    # audio_b64, answer = answer_question_v2(qna_dict, api_key, context, need_audio=True)
                    audio_b64, answer = answer_question_v2_langchain(qna_dict, api_key, context, need_audio=True)
                    data_dict = {'answer': answer, 'b64_of_answer': audio_b64}
                    return jsonify(data_dict)
                
                elif (no_context == False) and (int(need_audio) == 0):
                    # answer = answer_question_v2(qna_dict, api_key, context, need_audio=False)
                    answer = answer_question_v2_langchain(qna_dict, api_key, context, need_audio=False)
                    data_dict = {'answer': answer}
                    return jsonify(data_dict)
            except AuthenticationError:
                return jsonify('OpenAI API Key failed to authenticate.')
        elif (request.method == "GET"):
            return get_text
    except Exception as e:
        reference_number = generate_reference_number(8)
        logger.error(f"Error in /q2a_test route [Ref: {reference_number}]: \n{e}\n")
        return jsonify(f'Error occurred. Please check server log with Ref No. {reference_number}\nError Summary: {e}')
    
    

@app.route('/img2text', methods=['POST', 'GET'])
def img2text():

    if not os.path.exists("img2text_counter.txt"):
        with open("img2text_counter.txt", "w") as f:
            f.write("0")

    with open("img2text_counter.txt", "r") as f:
        counter = int(f.read())

    with open("img2text_counter.txt", "w") as f:
        f.write(str(counter + 1))

    try:
        if (request.method == "POST"):
            b64_img = request.get_json().get('b64_img')
            # text = img_to_text(b64_img)
            text = img_to_text_aws_textract(b64_img)
            return jsonify(text)
        elif (request.method == "GET"):
            return get_text
    except Exception as e:
        reference_number = generate_reference_number(8)
        logger.error(f"Error in /img2text route [Ref: {reference_number}]: \n{e}\n")
        return jsonify(f'Error occurred. Please check server log with Ref No. {reference_number}\nError Summary: {e}')
    

@app.route('/sentiment', methods=['POST', 'GET'])
def sentiment():
    try:

        if not os.path.exists("sentiment_counter.txt"):
            with open("sentiment_counter.txt", "w") as f:
                f.write("0")

        with open("sentiment_counter.txt", "r") as f:
            counter = int(f.read())

        with open("sentiment_counter.txt", "w") as f:
            f.write(str(counter + 1))

        if (request.method == "POST"):
            passage = request.get_json().get('passage')
            api_key = request.get_json().get('api_key')
            try:
                sentiment = sentiment_analysis(passage, api_key)
                return jsonify(sentiment)
            except AuthenticationError:
                return jsonify('OpenAI API Key failed to authenticate.')
            except Exception as e:
                return jsonify(str(e))
    except Exception as e:
        reference_number = generate_reference_number(8)
        logger.error(f"Error in /sentiment route [Ref: {reference_number}]: \n{e}\n")
        return jsonify(f'Error occurred. Please check server log with Ref No. {reference_number}\nError Summary: {e}')
        

        
@app.route('/text2audio', methods=['POST', 'GET'])
def text2audio():
    try:

        if not os.path.exists("text2audio_counter.txt"):
            with open("text2audio_counter.txt", "w") as f:
                f.write("0")

        with open("text2audio_counter.txt", "r") as f:
            counter = int(f.read())

        with open("text2audio_counter.txt", "w") as f:
            f.write(str(counter + 1))

        if (request.method == "POST"):
            text = request.get_json().get('text')
            audio = generate_audio_b64(text)
            return jsonify(audio)
    except Exception as e:
        reference_number = generate_reference_number(8)
        logger.error(f"Error in /text2audio route [Ref: {reference_number}]: \n{e}\n")
        return jsonify(f'Error occurred. Please check server log with Ref No. {reference_number}\nError Summary: {e}')
    

@app.route('/generate_pdf_id', methods=['POST', 'GET'])
def generate_pdf_id():
    try:

        if not os.path.exists("generate_pdf_id_counter.txt"):
            with open("generate_pdf_id_counter.txt", "w") as f:
                f.write("0")

        with open("generate_pdf_id_counter.txt", "r") as f:
            counter = int(f.read())

        with open("generate_pdf_id_counter.txt", "w") as f:
            f.write(str(counter + 1))

        if (request.method == "POST"):
            pdf_b64 = request.get_json().get('pdf_b64')
            pdf_id = save_pdf_file(pdf_b64)
            return jsonify(pdf_id)
    except Exception as e:
        reference_number = generate_reference_number(8)
        logger.error(f"Error in /generate_pdf_id route [Ref: {reference_number}]: \n{e}\n")
        return jsonify(f'Error occurred. Please check server log with Ref No. {reference_number}\nError Summary: {e}')
    

@app.route('/chat_pdf', methods=['POST', 'GET'])
def pdf_chat():
    try:

        if not os.path.exists("chat_pdf_counter.txt"):
            with open("chat_pdf_counter.txt", "w") as f:
                f.write("0")

        with open("chat_pdf_counter.txt", "r") as f:
            counter = int(f.read())

        with open("chat_pdf_counter.txt", "w") as f:
            f.write(str(counter + 1))

        if (request.method == "POST"):
            api_key = request.get_json().get('api_key')
            pdf_id = request.get_json().get('pdf_id')
            ques = request.get_json().get('question')
            response = chat_pdf(api_key, pdf_id, ques)
            return jsonify(response)
        
    except Exception as e:
        reference_number = generate_reference_number(8)
        logger.error(f"Error in /chat_pdf route [Ref: {reference_number}]: \n{e}\n")
        return jsonify(f'Error occurred. Please check server log with Ref No. {reference_number}\nError Summary: {e}')



if __name__ == '__main__':
    # app.run("127.0.0.2", 5000)
    app.run('0.0.0.0', 5000)