import chatbot_gpt4all as cbg4a
import chatbot_hf as cba
import chatbot_basics as basics
import chatbot_tf as cbtf
import chatbot_internet as cbi
import chatbot_pt as cbpt
import text_classification as tc
import translation as tns
import ai_generate_image as agi
import ai_img_to_text as itt
import ai_generate_audio as aga
import ai_filters as filters
import ai_depth_estimation as de
import ai_object_detection as od
import speech_recog as sr
import ai_config as cfg
import ai_conversation as conv
import ai_logs as logs
import PIL.Image as Image
import openai
import os
import datetime
import calendar
import base64
import json

"""
README

Models:
g4a - GPT4All chatbot (Text Generation).
tf - Tensorflow (selected version) chatbot (Text Generation).
cgpt - ChatGPT (API Key required) chatbot (Text Generation).
hf - Hugging Face selected model (Text Generation).
sc - Text classification (Sequence Classification).
tr - Translation.
int - Internet chatbot (Question Answering).
pt - PyTorch chatbot (Text Generation).
text2img - Image generation.
img2text - Image to Text.
whisper - Recognize audio using Whisper (Speech To Text).
text2audio - Audio generation.
nsfw_filter-text - Check if a text is NSFW.
nsfw_filter-image - Check if an image is NSFW.
od - Object Detection.
"""

openai_model: str = cfg.current_data.openai_model
openai_max_tokens: int = cfg.current_data.max_length
system_messages: list[str] = []
apply_system_messages_to_tensorflow: bool = False
use_chat_history_if_available: bool = cfg.current_data.use_chat_history
order: str = cfg.current_data.prompt_order
model_pt_name: str = "pt_model"
current_emotion: str = "happy"

def SystemMessagesToStr(extra_system_messages: list[str] = None) -> str:
    sm_str = "("

    if (cfg.current_data.use_default_system_messages):
        for i in system_messages:
            sm_str += i

        if (extra_system_messages != None):
            for i in extra_system_messages:
                sm_str += i
    
    sm_str += ") "
    return sm_str

def GetAllModels() -> dict[str]:
    models = order.split(" ")
    mls = {}

    for i in models:
        if (i == "g4a"):
            mls[i] = cfg.current_data.gpt4all_model
        elif (i == "tf"):
            mls[i] = "Custom TF Model"
        elif (i == "cgpt"):
            mls[i] = "ChatGPT"
        elif (i == "hf"):
            mls[i] = cfg.current_data.hf_model
        elif (i == "sc"):
            mls[i] = cfg.current_data.text_classification_model
        elif (i == "tr"):
            mls[i] = [cfg.current_data.translation_model_multiple] + list(cfg.current_data.translation_models.values())
        elif (i == "int"):
            mls[i] = cfg.current_data.internet_model
        elif (i == "pt"):
            mls[i] = "Custom PT Model"
        elif (i == "text2img"):
            mls[i] = cfg.current_data.image_generation_model
        elif (i == "img2text"):
            mls[i] = cfg.current_data.img_to_text_model
        elif (i == "text2audio"):
            mls[i] = cfg.current_data.text_to_audio_model
        elif (i == "whisper"):
            mls[i] = cfg.current_data.whisper_model
        elif (i == "nsfw_filter-text"):
            mls[i] = cfg.current_data.nsfw_filter_text_model
        elif (i == "nsfw_filter-image"):
            mls[i] = cfg.current_data.nsfw_filter_image_model
        elif (i == "depth"):
            mls[i] = cfg.current_data.depth_estimation_model
        elif (i == "od"):
            mls[i] = cfg.current_data.object_detection_model
    
    return mls

def LoadAllModels() -> None:
    models = order.split(" ")

    for i in models:
        if (i == "g4a"):
            cbg4a.LoadModel()
        elif (i == "tf"):
            cbtf.train_model()
        elif (i == "cgpt"):
            if (not os.path.exists("openai_api_key.txt")):
                with open("openai_api_key.txt", "w+") as oak:
                    oak.close()
        elif (i == "hf"):
            cba.LoadModel()
        elif (i == "sc"):
            tc.LoadModel()
        elif (i == "tr"):
            tns.LoadModels()
        elif (i == "int"):
            cbi.LoadModel()
        elif (i == "pt"):
            cbpt.TrainModel(model_pt_name)
        elif (i == "text2img"):
            agi.LoadModel()
        elif (i == "img2text"):
            itt.LoadModel()
        elif (i == "text2audio"):
            aga.LoadModel()
        elif (i == "nsfw_filter-text"):
            filters.LoadTextModel()
        elif (i == "nsfw_filter-image"):
            filters.LoadImageModel()
        elif (i == "depth"):
            de.LoadModel()
        elif (i == "od"):
            od.LoadModel()

def IsTextNSFW(prompt: str) -> bool:
    if (cfg.current_data.prompt_order.__contains__("nsfw_filter-text")):
        return filters.IsTextNSFW(prompt)
    
    return None

def IsImageNSFW(image: str | Image.Image) -> bool:
    if (cfg.current_data.prompt_order.__contains__("nsfw_filter-image")):
        return filters.IsImageNSFW(image)
    
    return None

def MakePrompt(prompt: str, order_prompt: list[str] = [], args: str = "", extra_system_msgs: list[str] = [], translator: str = "", force_translator: bool = True, conversation: list[str] = ["", ""], use_default_sys_prompts: bool = True) -> dict:
    global order, current_emotion
    LoadAllModels()

    sm = []

    if (cfg.current_data.use_default_system_messages):
        sm = system_messages if (use_default_sys_prompts) else []

    for msg in extra_system_msgs:
        sm.append(msg)

    if (len(cfg.current_data.custom_system_messages.strip()) > 0):
        sm.append(cfg.current_data.custom_system_messages)

    if (cfg.current_data.use_dynamic_system_args):
        current_dt = datetime.datetime.now()
        dsm = []

        dsm.append("Your current state of humor is '" + current_emotion + "'.")
        dsm.append("The current date is "
            + str(current_dt.day) + " of "
            + calendar.month_name[current_dt.month] + " "
            + str(current_dt.year)
            + ", and the current time is "
            + ("0" + str(current_dt.hour) if current_dt.hour < 10 else str(current_dt.hour)) + ":"
            + ("0" + str(current_dt.minute) if current_dt.minute < 10 else str(current_dt.minute)) + ".")

        for m in dsm:
            sm.append(m)
    
    if (cfg.current_data.system_messages_in_first_person):
        sm = basics.ToFirstPerson(sm)
    
    if (order_prompt.__contains__("cgpt")):
        openai_client = openai.Client()

        with open("openai_api_key.txt", "r") as oak:
            openai_client.api_key = oak.read()
            openai.api_key = oak.read()
            
            oak.close()
    
    for sp in sm:
        if (len(sp.split()) == 0):
            sm.remove(sp)
        elif (sp.endswith("\n")):
            sm[sm.index(sp)] = sp[:-1]
    
    cbg4a.system_messages = sm
    cba.system_messages = sm
    openai_msgs = [{"role": "system", "content": msg} for msg in sm] + [{"role": "user", "content": prompt}]

    if (len(order_prompt) <= 0):
        order_prompt = order

        if (len(order) <= 0):
            order_prompt = cfg.current_data.prompt_order
            order = order_prompt
    
    if (order_prompt.__contains__("tr") and force_translator):
        prompt = tns.TranslateFrom1To2(prompt)
    
    if (order_prompt.__contains__("nsfw_filter-text")):
        is_nsfw = IsTextNSFW(prompt)

        if (is_nsfw):
            if (not cfg.current_data.allow_processing_if_nsfw):
                return {
                    "response": "ERROR",
                    "model": "-1",
                    "files": {},
                    "tested_models": [],
                    "text_classification": "-1",
                    "title": "NO TITLE",
                    "errors": ["NSFW detected! NSFW is not allowed.", "NSFW"]
                }
    
    if (args.__contains__("-ncb")):
        data = {
            "response": "",
            "model": "-1",
            "files": {},
            "tested_models": [],
            "text_classification": "-1",
            "title": "NO TITLE",
            "errors": []
        }

        if (len(translator.strip()) > 0 and order_prompt.__contains__("tr")):
            data["response"] = tns.TranslateFrom2To1(prompt, translator)
            data["tested_models"].append("tr")

        if (args.__contains__("-img") and order_prompt.__contains__("text2img")):
            data["files"]["image"] = GenerateImage(prompt)
            data["response"] = "[agi " + prompt + "]"
            data["tested_models"].append("text2img")
        
        if (args.__contains__("-aud") and order_prompt.__contains__("text2audio")):
            data["files"]["audio"] = GenerateAudio(prompt)
            data["response"] = "[aga " + prompt + "]"
            data["tested_models"].append("text2audio")
        
        if (args.__contains__("-depth") and order_prompt.__contains__("depth")):
            is_nsfw = IsImageNSFW(prompt)

            if (is_nsfw):
                if (not cfg.current_data.allow_processing_if_nsfw):
                    return {
                        "response": "ERROR",
                        "model": "-1",
                        "files": {},
                        "tested_models": [],
                        "text_classification": "-1",
                        "title": "NO TITLE",
                        "errors": ["NSFW detected! NSFW is not allowed.", "NSFW"]
                    }

            data["files"]["image"] = EstimateDepth(prompt)
            data["response"] = "[de " + prompt + "]"
            data["tested_models"].append("depth")
        
        if (args.__contains__("-od") and order_prompt.__contains__("od")):
            is_nsfw = IsImageNSFW(prompt)

            if (is_nsfw):
                if (not cfg.current_data.allow_processing_if_nsfw):
                    return {
                        "response": "ERROR",
                        "model": "-1",
                        "files": {},
                        "tested_models": [],
                        "text_classification": "-1",
                        "title": "NO TITLE",
                        "errors": ["NSFW detected! NSFW is not allowed.", "NSFW"]
                    }
                
            od = DetectObjects(prompt)
            data["files"]["image"] = od["image"]
            data["response"] = str(od["objects"])
            data["tested_models"].append("od")
        
        data["files"] = str(data["files"])
        return data
    
    tested_models: list[str] = []
    errors: list[str] = []
    responses = []

    for i in order_prompt.split(" "):
        response = "ERROR"

        try:
            if (i == "g4a"):
                response = "### RESPONSE: " + cbg4a.MakePrompt(prompt, use_chat_history_if_available, conversation)
            elif (i == "tf"):
                response = "### RESPONSE: " + cbtf.get_ai_response((SystemMessagesToStr() if apply_system_messages_to_tensorflow else "") + prompt)
            elif (i == "cgpt"):
                response = "### RESPONSE: " + openai_client.chat.completions.create(model = openai_model, messages = openai_msgs, temperature = cfg.current_data.temp, max_tokens = openai_max_tokens).choices[0].message
            elif (i == "hf"):
                response = "### RESPONSE: " + cba.MakePrompt(prompt, use_chat_history_if_available, conversation)
            elif (i == "int"):
                response = "### RESPONSE: " + cbi.MakePrompt(prompt)
            elif (i == "pt"):
                response = "### RESPONSE: " + cbpt.GenerateResponse((SystemMessagesToStr() if apply_system_messages_to_tensorflow else "") + prompt)
            elif (i == "img2text"):
                if (order_prompt.__contains__("nsfw_filter-image")):
                    is_nsfw = IsImageNSFW(prompt)

                    if (is_nsfw):
                        if (not cfg.current_data.allow_processing_if_nsfw):
                            return {
                                "response": "ERROR",
                                "model": "-1",
                                "files": {},
                                "tested_models": [],
                                "text_classification": "-1",
                                "title": "NO TITLE",
                                "errors": ["NSFW detected! NSFW is not allowed.", "NSFW"]
                            }
            
                response = "### RESPONSE: " + ImageToText(prompt)
            
            tested_models.append(i)
        except Exception as ex:
            errors.append("Error on model '" + i + "': '" + str(ex) + "'")
            logs.AddToLog("Model '" + i + "' found an error processing '" + prompt + "': '" + str(ex) + "'")

            continue

        response = response.strip()
        
        if (response.startswith("### RESPONSE: ")):
            response = response[14:len(response)]

        try:
            if (response.__contains__("### USER")):
                response = response.split("### USER")[0]

            if (response.__contains__("### Explanation")):
                response = response.split("### Explanation")[0]
            
            if (response.__contains__("### Prompt")):
                response = response.split("### Prompt")[0]

            if (response.__contains__("### Task")):
                response = response.split("### Task")[0]

            if (response.__contains__("### ASSISTANT")):
                response = response.split("### ASSISTANT")[0]
        except:
            pass
            
        if (len(response.strip()) > 0 and response.lower() != "error" and response.lower() != "error."):
            tcn: str = "-1"
            files: dict[str, str] = {}

            logs.AddToLog("Model '" + i + "' response to '" + prompt + "': '" + response + "'")

            if (cfg.current_data.allow_titles and response.startswith("[") and response.__contains__("]")):
                title = response[1:response.index("]")]
                response = response[response.index("]") + 1:len(response)].strip()

                if (response.startswith(" ")):
                    response = response[1:len(response)]

                if (len(response.strip()) == 0):
                    response = "[" + title + "]"
            else:
                title = "NO TITLE"
            
            if (cfg.current_data.use_multi_model):
                responses.append(response)
                continue

            if (cfg.current_data.save_conversations):
                if (cfg.current_data.allow_titles and response.startswith("[") and response.__contains__("]")):
                    response_conv = response[response.index("]") + 1:len(response)].strip()

                    if (response_conv.startswith(" ")):
                        response_conv = response_conv[1:len(response_conv)]
                else:
                    response_conv = response

                if (len(response_conv.strip()) == 0):
                    response_conv = response
                
                response_conv = response_conv.strip()
                conv.AddToConversation(conversation[0], conversation[1], prompt, response_conv)

            if (order_prompt.__contains__("sc")):
                try:
                    tcn = tc.DoPrompt(response)
                except Exception as ex:
                    tcn = "-1"
            
            if (order_prompt.__contains__("text2img") and response.__contains__("[agi ")):
                try:
                    img_prompt = response[response.index("[agi ") + 5:response[response.index("[agi ") + 5:len(response)].index("]")]
                    response = response.replace("[agi " + img_prompt + "]", "")
                    files["image"] = GenerateImage(img_prompt)

                    tested_models.append("text2img")
                except Exception as ex:
                    errors.append("Could not generate image: " + str(ex))
            
            if (order_prompt.__contains__("text2audio") and response.__contains__("[aga ")):
                try:
                    aud_prompt = response[response.index("[aga ") + 5:response[response.index("[aga ") + 5:len(response)].index("]")]
                    response = response.replace("[aga " + aud_prompt + "]", "")
                    files["audio"] = GenerateAudio(aud_prompt)

                    tested_models.append("text2audio")
                except Exception as ex:
                    errors.append("Could not generate audio: " + str(ex))
            
            if (order_prompt.__contains__("tr") and len(translator.strip()) > 0):
                response = tns.TranslateFrom2To1(response, translator)
            
            if (cfg.current_data.use_dynamic_system_args):
                for m in dsm:
                    if (sm.count(m) > 0):
                        sm.remove(m)
            
            return {
                "response": response,
                "model": i,
                "files": str(files),
                "tested_models": tested_models,
                "text_classification": tcn,
                "title": title,
                "errors": errors
            }
    
    if (cfg.current_data.use_multi_model):
        response = ""

        if (cfg.current_data.multi_model_mode == "shortest"):
            for r in responses:
                if (len(response.strip()) == 0):
                    response = r
                    continue

                if (len(r) < len(response)):
                    response = r
        elif (cfg.current_data.multi_model_mode == "longest"):
            for r in responses:
                if (len(response.strip()) == 0):
                    response = r
                    continue

                if (len(r) > len(response)):
                    response = r

        if (cfg.current_data.allow_titles and response.startswith("[") and response.__contains__("]")):
            title = response[1:response.index("]")]
            response = response[response.index("]") + 1:len(response)].strip()

            if (response.startswith(" ")):
                response = response[1:len(response)]

            if (len(response.strip()) == 0):
                response = "[" + title + "]"
        else:
            title = "NO TITLE"

        if (cfg.current_data.save_conversations):
            if (cfg.current_data.allow_titles and response.startswith("[") and response.__contains__("]")):
                response_conv = response[response.index("]") + 1:len(response)].strip()

                if (response_conv.startswith(" ")):
                    response_conv = response_conv[1:len(response_conv)]
            else:
                response_conv = response

            if (len(response_conv.strip()) == 0):
                response_conv = response

            response_conv = response_conv.strip()
            conv.AddToConversation(conversation[0], conversation[1], prompt, response_conv)
        
        if (order_prompt.__contains__("sc")):
                try:
                    tcn = tc.DoPrompt(response)
                except Exception as ex:
                    tcn = "-1"
            
        if (order_prompt.__contains__("text2img") and response.__contains__("[agi ")):
            try:
                img_prompt = response[response.index("[agi ") + 5:response[response.index("[agi ") + 5:len(response)].index("]")]
                response = response.replace("[agi " + img_prompt + "]", "")
                files["image"] = GenerateImage(img_prompt)

                tested_models.append("text2img")
            except Exception as ex:
                errors.append("Could not generate image: " + str(ex))
        
        if (order_prompt.__contains__("text2audio") and response.__contains__("[aga ")):
                try:
                    aud_prompt = response[response.index("[aga ") + 5:response[response.index("[aga ") + 5:len(response)].index("]")]
                    response = response.replace("[aga " + aud_prompt + "]", "")
                    files["audio"] = GenerateAudio(aud_prompt)

                    tested_models.append("text2audio")
                except Exception as ex:
                    errors.append("Could not generate audio: " + str(ex))
            
        if (order_prompt.__contains__("tr") and len(translator.strip()) > 0):
            response = tns.TranslateFrom2To1(response, translator)
        
        return {
            "response": response,
            "files": str(files),
            "model": "-1",
            "tested_models": tested_models,
            "text_classification": tcn,
            "title": title,
            "errors": errors
        }

    return {
        "response": "ERROR",
        "files": str({}),
        "model": "-1",
        "tested_models": tested_models,
        "text_classification": "-1",
        "title": "NO TITLE",
        "errors": ["Error on response from all models."] + errors
    }

def SaveTF(model_name_tf: str = "tf_model", model_name_pt: str = "pt_model") -> None:
    if (cfg.current_data.prompt_order.__contains__("1")):
        cbtf.save_model(model_name_tf)
    
    if (cfg.current_data.prompt_order.__contains__("7")):
        cbpt.SaveModel(model_name_pt)

def GenerateImage(prompt: str) -> str:
    img_response = agi.GenerateImage(prompt)
    image = base64.b64encode(img_response).decode("utf-8")

    return image

def EstimateDepth(img: str) -> str:
    img_response = de.EstimateDepth(img)
    image = base64.b64encode(img_response).decode("utf-8")

    return image

def GenerateAudio(prompt: str) -> str:
    aud_response = aga.GenerateAudio(prompt)
    audio = base64.b64encode(aud_response).decode("utf-8")

    return audio

def ImageToText(img: str) -> str:
    return itt.MakePrompt(img)

def RecognizeAudio(audio: str) -> str:
    return sr.Recognize(sr.FileToAudioData(audio))

def DetectObjects(img: str) -> dict[str]:
    data = od.MakePrompt(img)
    image = base64.b64encode(data["image"]).decode("utf-8")

    return {
        "objects": data["objects"],
        "image": image
    }