from openai import OpenAI
client = OpenAI()
import vertexai
from vertexai.vision_models import ImageTextModel, Image
from google.cloud import vision
import os
import tempfile
import gradio as gr
from cryptography.fernet import Fernet
import concurrent.futures
import proto
import json
from deep_translator import GoogleTranslator
from vertexai.preview.language_models import TextGenerationModel
from jose import JWTError, jwt
from dotenv import load_dotenv
import time
from transformers import BlipProcessor, BlipForConditionalGeneration
import io
import boto3
from PIL import Image as PIM
import requests
import warnings
import base64
warnings.filterwarnings("ignore")
alt,desc,lang,obs,cap,img,llm = "","","","","","",""
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import concurrent.futures

language_codes = {'Afrikaans': 'af','Albanian': 'sq','Amharic': 'am','Arabic': 'ar','Armenian': 'hy','Assamese': 'as','Aymara': 'ay','Azerbaijani': 'az','Bambara': 'bm','Basque': 'eu','Belarusian': 'be','Bengali': 'bn','Bhojpuri': 'bho','Bosnian': 'bs','Bulgarian': 'bg','Catalan': 'ca','Cebuano': 'ceb','Chinese': 'zh-TW','Corsican': 'co','Croatian': 'hr','Czech': 'cs','Danish': 'da','Dhivehi': 'dv','Dogri': 'doi','Dutch': 'nl','English': 'en','Esperanto': 'eo','Estonian': 'et','Ewe': 'ee','Filipino': 'fil','Tagalog': 'fil','Finnish': 'fi','French': 'fr','Frisian': 'fy','Galician': 'gl','Georgian': 'ka','German': 'de','Greek': 'el','Guarani': 'gn','Gujarati': 'gu','Haitian Creole': 'ht','Hausa': 'ha','Hawaiian': 'haw','Hebrew': 'he','Hindi': 'hi','Hmong': 'hmn','Hungarian': 'hu','Icelandic': 'is','Igbo': 'ig','Ilocano': 'ilo','Indonesian': 'id','Irish': 'ga','Italian': 'it','Japanese': 'ja','Javanese': 'jv','Kannada': 'kn','Kazakh': 'kk','Khmer': 'km','Kinyarwanda': 'rw','Konkani': 'gom','Korean': 'ko','Krio': 'kri','Kurdish': 'ku','Kurdish': 'ckb','Sorani': 'ckb','Kyrgyz': 'ky','Lao': 'lo','Latin': 'la','Latvian': 'lv','Lingala': 'ln','Lithuanian': 'lt','Luganda': 'lg','Luxembourgish': 'lb','Macedonian': 'mk','Maithili': 'mai','Malagasy': 'mg','Malay': 'ms','Malayalam': 'ml','Maltese': 'mt','Maori': 'mi','Marathi': 'mr','Meiteilon': 'mni-Mtei','Manipuri': 'mni-Mtei','Mizo': 'lus','Mongolian': 'mn','Myanmar': 'my','Burmese': 'my','Nepali': 'ne','Norwegian': 'no','Nyanja': 'ny','Chichewa':'ny','Odia': 'or','Oriya': 'or','Oromo': 'om','Pashto': 'ps','Persian': 'fa','Polish': 'pl','Portuguese': 'pt','Punjabi': 'pa','Quechua': 'qu','Romanian': 'ro','Russian': 'ru','Samoan': 'sm','Sanskrit': 'sa','Scots Gaelic': 'gd','Sepedi': 'nso','Serbian': 'sr','Sesotho': 'st','Shona': 'sn','Sindhi': 'sd','Sinhala': 'si','Slovak': 'sk','Slovenian': 'sl','Somali': 'so','Spanish': 'es','Sundanese': 'su','Swahili': 'sw','Swedish': 'sv','Tagalog': 'tl','Tajik': 'tg','Tamil': 'ta','Tatar': 'tt','Telugu': 'te','Thai': 'th','Tigrinya': 'ti','Tsonga': 'ts','Turkish': 'tr','Turkmen': 'tk','Twi': 'ak','Ukrainian': 'uk','Urdu': 'ur','Uyghur': 'ug','Uzbek': 'uz','Vietnamese': 'vi','Welsh': 'cy','Xhosa': 'xh','Yiddish': 'yi','Yoruba': 'yo','Zulu': 'zu'}

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

encryption_key = os.environ['enc_key']

fernet = Fernet(encryption_key)

rekognition = boto3.client(
    'rekognition',
    region_name='us-east-1',
    aws_access_key_id=os.environ['aws_access_key'],
    aws_secret_access_key=os.environ['aws_secret_access']
)

BLIPprocessor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
BLIPmodel = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Decrypt the encrypted data
with open("encrypted_data.json", "rb") as encrypted_file:
    encrypted_data = encrypted_file.read()
    decrypted_data = fernet.decrypt(encrypted_data)

with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
    temp_file.write(decrypted_data)
    temp_file_path = temp_file.name

# Set the temporary file path as the Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path

PROJECT_ID = os.environ['project_id']
LOCATION = 'us-central1'
vertexai.init(project=PROJECT_ID, location=LOCATION)
VISIONmodel = ImageTextModel.from_pretrained("imagetext@001")
VISIONclient = vision.ImageAnnotatorClient()

# GPT4 to generate descriptions
def GPT4des(para):
    s = time.time()
    feeder = (
        "Generate a comprehensive description of the image's content for its alt text. Pay attention to available details and emotions conveyed. Avoid personal analysis and keep the description concise.. The details are as follows - \n"
        + para
    )
    formatter = "Craft a coherent image description that encompasses all available information. Prioritize accuracy and coherence without adding extra details. Be creative while maintaining simplicity. Aim for a description of around 100 words without including additional information."
    curator = "When describing a poster, exercise caution, particularly when posters convey deep emotions. Differentiate between objects and the conveyed emotions. Avoid redundant phrases like 'This message appears multiple times' and concentrate on capturing the essence of the content and emotions conveyed."

    instruction = "start with 'A picture of '"

    conversation = [
        {"role": "system", "content": "You are an AI language model."},
        {"role": "user", "content": feeder},
        {"role": "user", "content": formatter},
        {"role": "user", "content": curator},
        {"role": "user", "content": instruction},
    ]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=conversation,
    )

    # print("desc - ", time.time() - s)
    return response.choices[0].message.content

# PaLM2 to generate descriptions
def PaLM2des(
    project_id: str,
    model_name: str,
    temperature: float,
    max_decode_steps: int,
    top_p: float,
    top_k: int,
    content: str,
    location: str = "us-central1",
    tuned_model_name: str = "",
):
    infoString = [
        {"role": "system", "content": "You are an AI language model."},
        {
            "role": "user",
            "content": "Generate a comprehensive description of the image's content for its alt text. Pay attention to available details and emotions conveyed. Avoid personal analysis and keep the description concise.. The details are as follows - \n"
            + content,
        },
        {
            "role": "user",
            "content": "Craft a coherent image description that encompasses all available information. Prioritize accuracy and coherence without adding extra details. Be creative while maintaining simplicity. Aim for a description of around 100 words without including additional information.",
        },
        {
            "role": "user",
            "content": "When describing a poster, exercise caution, particularly when posters convey deep emotions. Differentiate between objects and the conveyed emotions. Avoid redundant phrases like 'This message appears multiple times' and concentrate on capturing the essence of the content and emotions conveyed.",
        },
        {"role": "user", "content": "start with 'A picture of '"},
    ]

    content = "follow the instructions in this conversation structure below\n" + str(
        infoString
    )

    model = TextGenerationModel.from_pretrained(model_name)

    if tuned_model_name:
        model = model.get_tuned_model(tuned_model_name)

    response = model.predict(
        content,
        temperature=temperature,
        max_output_tokens=max_decode_steps,
        top_k=top_k,
        top_p=top_p,
    )

    return response.text

# Llama2 to generate descriptions
def Llama2des(text):
    pplx_api_key=os.environ['pplx_api_key']
    payload = {
        "model": "llama-2-70b-chat",
        "messages": [
            {"role": "system", "content": "You are an AI language model."},
            {
                "role": "user",
                "content": "Generate a comprehensive description of the image's content for its alt text. Pay attention to available details and emotions conveyed. Avoid personal analysis and keep the description concise.. The details are as follows - \n"
                + text,
            },
            {
                "role": "user",
                "content": "Craft a coherent image description that encompasses all available information. Prioritize accuracy and coherence without adding extra details. Be creative while maintaining simplicity. Aim for a description of around 100 words without including additional information.",
            },
            {
                "role": "user",
                "content": "When describing a poster, exercise caution, particularly when posters convey deep emotions. Differentiate between objects and the conveyed emotions. Avoid redundant phrases like 'This message appears multiple times' and concentrate on capturing the essence of the content and emotions conveyed.",
            },
            {"role": "user", "content": "Strictly start with 'A picture of '"},
        ],
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {pplx_api_key}",
    }

    response = requests.post(
        "https://api.perplexity.ai/chat/completions", json=payload, headers=headers
    ).json()

    return response["choices"][0]["message"]["content"]

def gpt4V(path):
    key=os.environ['OPENAI_API_KEY']
    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # Getting the base64 string
    base64_image = encode_image(path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Write a caption and a 100-word description for this image by trying to answer the following questions:
                1. Who are the people in this picture?
                2. What objects or subjects does this image contain?
                3. Extract all the text and numbers from this image.
                4. Describe the events or activities happening in this image.
                5. If applicable, describe the emotions portrayed in the image.

                ** If any specific information is missing, please naturally incorporate the available details into your response. Avoid stating the absence of certain elements explicitly; instead, prioritize creating a comprehensive and engaging narrative around the visible content. Ensure that the description is grammatically correct and coherent.                
                If specific details are not visible/not depicted, seamlessly integrate the available information into your narrative without explicitly stating the absence of certain elements.
                Strictly Avoid saying "There are No x, y, z present/depicited in image." **
                
                Please provide a JSON response in the following format - 
                {
                    "altText": "Enter a suitable title or alt text for the image",
                    "Description": "Compose a detailed description of the image content with a 100-word limit."
                }
                In your response, feel free to omit any information that is not applicable or missing from the image.""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": 4096,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    ).json()
    # print(response)

    string = response["choices"][0]["message"]["content"]
    # print(string)
    a=json.loads("{" + str(string.split("{")[-1].split("}")[0]) + "}")
    return a['altText'],a['Description']

# check for obscenity in images and flag the ones which are NSFW
def gcVision(content):  # , temp_image_path):
    s = time.time()
    image = vision.Image(content=content)
    response = VISIONclient.safe_search_detection(image=image)
    safe = response.safe_search_annotation
    # print(safe)
    safe = proto.Message.to_json(safe)
    safe = json.loads(safe)
    safe["spoof"] = 0

    # print("safe - ", time.time() - s)
    # print(safe)
    return any(
        value > 3 for value in list(safe.values())
    )  # or rekog["ModerationLabels"]


# AWS Rekognition module for obscenity check
def AWSrekognition(path):
    s = time.time()
    # Read the image file as bytes
    with open(path, "rb") as image_file:
        image_bytes = image_file.read()
    try:
        response = rekognition.detect_moderation_labels(
            Image={"Bytes": image_bytes}, MinConfidence=75
        )
        # print(response)
        print("rekognition - ", time.time() - s)
        if response["ModerationLabels"]:
            return True
        else:
            return False
    except Exception as e:
        print(e)

# ask questions on the image and return one answer at a time
def get_answers(image, question):
    s = time.time()
    answers = VISIONmodel.ask_question(
        image=image, question=question, number_of_results=3
    )  # Get 3 results
    valid_answers = [
        ans
        for ans in answers
        if ans
        not in (
            "unanswered",
            "no text in image",
            "no",
            "unanswerable",
            "no text",
            "unsuitable",
        )
    ]
    if valid_answers:
        # print("getans - ", time.time() - s)
        return " ".join(list(set(valid_answers)))
    return []

# function to concurrently generate answers based on questions asked on images
def collect_answers(image, questions):
    s = time.time()
    all_answers = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(get_answers, image, question) for question in questions
        ]
        for future in concurrent.futures.as_completed(futures):
            ans = future.result()
            # print(ans)
            if ans:
                all_answers.append(ans)  # Add answers to the combined list
    # print("collectans - ", time.time() - s)
    # print(all_answers)
    return "".join(all_answers)

# Generating captions with google cloud vision
def VISIONcap(image_data, desModel):
    s = time.time()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image_file:
        temp_image_path = temp_image_file.name
        temp_image_file.write(image_data)

    try:
        image = Image.load_from_file(location=temp_image_path)
        questions = [
            "name the people in this picture.",
            "What does this image contain?",
            "Give me all the text and all the numbers in this image.",
            "What is happening in this image?",
            "what are the emotions being potrayed in the image? do not answer if not necessary.",
        ]

        # Create a ThreadPoolExecutor for concurrent execution
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit the collect_answers function to the executor
            answers_future = executor.submit(collect_answers, image, questions)

            # Use the model.get_captions function directly
            captions = VISIONmodel.get_captions(
                image=image, number_of_results=1, language="en"
            )

        # Get the results from the answers and captions futures
        answers = answers_future.result()
        # print(answers)
        alt = captions[0].capitalize()
        desString = (
            "\nAlt Text of the image - "
            + alt
            + "\nObjects and emotions in the picture - "
            + answers
        )
        des = genDescription(desString, desModel)
        # print(alt, des)
        # print("generate - ", time.time() - s)

        return alt, des

    except Exception as e:
        return e, e


def split_text(input_text, max_chunk_length=4995):
    sentences = sent_tokenize(input_text)

    # Initialize variables
    chunks = []
    current_chunk = ""

    # Iterate through sentences
    for sentence in sentences:
        # Check if adding the sentence to the current chunk exceeds the max length
        if len(current_chunk) + len(sentence) <= max_chunk_length:
            current_chunk += sentence + " "
        else:
            # Save the current chunk and start a new one with the current sentence
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def gtranslate(language, text):
    return GoogleTranslator(source="auto", target=language_codes[language]).translate(text)

def translate(text, language):
    if len(text)>=5000:
        print('Character length greater than 5000')
        result_chunks = split_text(text)
        language = [language]*len(result_chunks)
        with concurrent.futures.ThreadPoolExecutor() as executor:
          result_chunks = list(executor.map(gtranslate, language, result_chunks))
        return ' '.join(result_chunks)
    return GoogleTranslator(
                        source="auto", target=language_codes[language]
                    ).translate(text)

# Generation Captions with salesforce BLIP
def BLIPcaptions(raw_image, desModel):
    # conditional image captioning
    text = "A picture of"
    inputs = BLIPprocessor(
        raw_image, text, return_tensors="pt"
    )  # .to("cuda", torch.float16)

    out = BLIPmodel.generate(**inputs)

    return BLIPprocessor.decode(out[0], skip_special_tokens=True), genDescription(
        BLIPprocessor.decode(out[0], skip_special_tokens=True), desModel
    )

# Flow control function
def genDescription(infoString, desModel):
    if desModel == "Llama2":
        return Llama2des(infoString)
    elif desModel == "GPT4":
        return GPT4des(infoString)
    elif desModel == "PaLM2":
        return PaLM2des("data-science-401804","text-bison",0.2,512,0.8,40,infoString,"us-central1",)
    else:
        return "Unrecognized Description Provider"

def predict(image, language, obs_model, img_model, desc_model):
    print(image)
    print(language)
    print(obs_model)
    print(img_model)
    print(desc_model)
    global alt, desc, obs, llm, img, cap
    if (image==img and obs_model==obs and desc_model==llm and img_model==cap):
        with concurrent.futures.ThreadPoolExecutor() as executor:
                alt_future = executor.submit(translate, alt, language)
                des_future = executor.submit(translate, des, language)
        alt=alt_future.result()
        des=des_future.result()
        print(alt)
        print(des)
        return alt, des
    else:
        img=image
        obs=obs_model
        llm=desc_model
        cap=img_model
        with open(image, 'rb') as image_file:
            image_bytes = image_file.read()
        img = PIM.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")
    
        # Save the converted image back to bytes
        img_byte_array = io.BytesIO()
        img.save(img_byte_array, format="PNG")
    
        # Use the `getvalue()` method to retrieve the bytes-like object
        img_bytes = img_byte_array.getvalue()
    
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image_file:
            temp_image_path = temp_image_file.name
            temp_image_file.write(img_bytes)
    
        # Obscenity Providers
        if obs_model == "awsRekognition":
            if AWSrekognition(temp_image_path):
                return "Unsafe Image!", "Try again with a different Image!"
        elif obs_model == "gcVision":
            if gcVision(img_bytes):
                return "Unsafe Image!", "Try again with a different Image!"
        else:
            return "Incorrect Provider for obscenity check!", "Choose one of awsRekognition and gcVision!"
    
        # Caption Providers
        if img_model == "gcVertex":
            alt, des = VISIONcap(img_byte_array.getvalue(), desc_model)
        elif img_model == "sfBLIP":
            alt, des = BLIPcaptions(PIM.open(temp_image_path), desc_model)
        elif img_model == "gpt4V":
            alt, des = gpt4V(temp_image_path)
        else:
            return "Incorrect Provider for generating content!", "Choose one of gcVertex and sfBLIP!"
    
    
        # Language Translation
        if not language:
            return "Incorrect Language or Language not Available!","Choose one of the dropdown languages only!"
    
        elif language and language != "English":
            with concurrent.futures.ThreadPoolExecutor() as executor:
                alt_future = executor.submit(translate, alt, language)
                des_future = executor.submit(translate, des, language)
            alt=alt_future.result()
            des=des_future.result()
            print(alt)
            print(des)
            return alt, des
    
        else:
            print(alt)
            print(des)
            return alt, des

iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.File(label="Upload Image"),
        gr.Dropdown(['English', 'Afrikaans', 'Albanian', 'Amharic', 'Arabic', 'Armenian', 'Assamese', 'Aymara', 'Azerbaijani', 'Bambara', 'Basque', 'Belarusian', 'Bengali', 'Bhojpuri', 'Bosnian', 'Bulgarian', 'Catalan', 'Cebuano', 'Chinese', 'Corsican', 'Croatian', 'Czech', 'Danish', 'Dhivehi', 'Dogri', 'Dutch', 'Esperanto', 'Estonian', 'Ewe', 'Filipino', 'Tagalog', 'Finnish', 'French', 'Frisian', 'Galician', 'Georgian', 'German', 'Greek', 'Guarani', 'Gujarati', 'Haitian Creole', 'Hausa', 'Hawaiian', 'Hebrew', 'Hindi', 'Hmong', 'Hungarian', 'Icelandic', 'Igbo', 'Ilocano', 'Indonesian', 'Irish', 'Italian', 'Japanese', 'Javanese', 'Kannada', 'Kazakh', 'Khmer', 'Kinyarwanda', 'Konkani', 'Korean', 'Krio', 'Kurdish', 'Sorani', 'Kyrgyz', 'Lao', 'Latin', 'Latvian', 'Lingala', 'Lithuanian', 'Luganda', 'Luxembourgish', 'Macedonian', 'Maithili', 'Malagasy', 'Malay', 'Malayalam', 'Maltese', 'Maori', 'Marathi', 'Meiteilon', 'Manipuri', 'Mizo', 'Mongolian', 'Myanmar', 'Burmese', 'Nepali', 'Norwegian', 'Nyanja', 'Chichewa', 'Odia', 'Oriya', 'Oromo', 'Pashto', 'Persian', 'Polish', 'Portuguese', 'Punjabi', 'Quechua', 'Romanian', 'Russian', 'Samoan', 'Sanskrit', 'Scots Gaelic', 'Sepedi', 'Serbian', 'Sesotho', 'Shona', 'Sindhi', 'Sinhala', 'Slovak', 'Slovenian', 'Somali', 'Spanish', 'Sundanese', 'Swahili', 'Swedish', 'Tajik', 'Tamil', 'Tatar', 'Telugu', 'Thai', 'Tigrinya', 'Tsonga', 'Turkish', 'Turkmen', 'Twi', 'Ukrainian', 'Urdu', 'Uyghur', 'Uzbek', 'Vietnamese', 'Welsh', 'Xhosa', 'Yiddish', 'Yoruba', 'Zulu'], label="Select Language"),
        gr.Dropdown(["awsRekognition", "gcVision"], label="Select Obscenity Model"),
        gr.Dropdown(["gcVertex", "sfBLIP", "gpt4V"], label="Select Caption Provider Model"),
        gr.Dropdown(["GPT4", "Llama2", "PaLM2"], label="Select Text Generation Model"),
            ],
    outputs=[
        gr.Textbox(label="Alt Text"),
        gr.Textbox(label="Description")
    ],
    title="Alt Text and Description Generator",
    description="Generate alt text and description for an uploaded image using AI models."
)

iface.launch()