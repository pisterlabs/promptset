import cv2
import clip
from PIL import Image
import numpy as np
import random
import requests
from deepface import DeepFace
import torch
import openai


OPENAI_API_KEY = "Your OpenAI API Key"
openai.api_key = OPENAI_API_KEY


frontalface_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(frontalface_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def insert_str(original_str, pos, adding_str):
    str_list = list(original_str)
    str_list.insert(pos, adding_str)
    res_str = ''.join(str_list)
    return  res_str

def reframe_box(box, image_size):
    (width, height) = image_size
    delta = 150
    (x, y, w, h) = box
    x = max(0, x-delta)
    y = max(0, y-delta)
    w = min(width, w+(2*delta))
    h = min(height, h+(2*delta))
    return (x, y, w, h)

def get_cropped_face_image(image):
    faces = face_cascade.detectMultiScale(np.array(image), scaleFactor=1.1, minNeighbors=5, minSize=(200, 200))
    if (len(faces) == 0):
        return (False, None)
    else:
        a_face_box = random.choice(faces)
        (x, y, w, h) = reframe_box(a_face_box, image.size)
        cropped_image = image.crop((x, y, x+w, y+h))
        return (True, cropped_image)
    
def get_face_emotion_and_other_properties(image):
    face_analysis_list = DeepFace.analyze(img_path=np.array(image))
    if (len(face_analysis_list) == 0):
        return None
    face_analysis = face_analysis_list[0]
    # print("face_analysis:", face_analysis)
    res_dic = {}
    res_dic["dominant_emotion"] = face_analysis["dominant_emotion"].lower()
    res_dic["dominant_gender"] = face_analysis["dominant_gender"].lower()
    res_dic["dominant_race"] = face_analysis["dominant_race"].lower()
    res_dic["age"] = face_analysis["age"]
    return res_dic

def get_attractiveness_of_face(face_image, gender):
    cls_names = ["beautiful", "common", "ugly"]
    if (gender == "man"):
        cls_names = ["handsome", "common", "ugly"]

    image = preprocess(face_image).unsqueeze(0).to(device)
    text = clip.tokenize(cls_names).to(device)
    with torch.no_grad():
        logits_per_image, _ = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        print("probs:", probs)
        return (probs.argmax(), probs.max())

def generate_sacarstic_seed_by_attractiveness(attractiveness_index, attractiveness_score, gender, age):
    sacarstic_seed = ("a " + str(age) + "-year-old ")

    if (attractiveness_index == 0):
        face_adj = "beautiful"
        if (gender == "man"):
            face_adj = "handsome"

        if (attractiveness_score < 80):
            sacarstic_seed += ("relatively " + face_adj + " ")
        else:
            sacarstic_seed += (face_adj + " ")

        if (age < 20):
            sacarstic_seed += "young "
            if (gender == "man"):                
                sacarstic_seed += "boy"
            else:
                sacarstic_seed += "girl"
        else:
            if (age > 50):
                sacarstic_seed += "old "
            if (gender == "man"):                
                sacarstic_seed += "man"
            else:
                sacarstic_seed += "woman"
    return sacarstic_seed
        
def get_adjective_words(seed_sentense):
    prompt = f"What non-negative adjective words can be used to describe \"{seed_sentense}\"? List 5 of them and their antonym word in pure 2-d python array."
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.73,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    choices = response["choices"]
    answer_item = choices[0]
    adj_words_list_str = answer_item["text"].strip()
    # print("adj_words_list_str:", adj_words_list_str)
    adj_words_list = eval(adj_words_list_str)
    return adj_words_list

def get_analogy(seed_sentense, adj_w):
    seed_sen = insert_str(seed_sentense, 2, (adj_w + " "))
    prompt = f"What thing can be used as an analogy to \"{seed_sen}\"?"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # print("response:", response)
    choices = response["choices"]
    answer_item = choices[0]
    analogy = answer_item["text"].strip().lower()
    if (analogy.endswith(".")):
        analogy = analogy[:-1]
    return analogy

def paraphrase_by_attractiveness(gender, age, p_adj_w, n_adj_w, analogy):
    sentence = ""    
    if (gender == "man"):
        sentence += ("he is so " + p_adj_w + " ")
        if (age < 20):
            sentence += "boy"
        else:
            sentence += "man"
    else:
        sentence += ("she is so " + p_adj_w + " ")
        if (age < 20):
            sentence += "girl"
        else:
            sentence += "woman"
    sentence += (", like " + analogy)

    prompt = f"Paraphrase this sentence: \"{sentence}\", in another form, without transition words like \"but\", \"yet\", etc."
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.91,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    choices = response["choices"]
    answer_item = choices[0]
    paraphrased_sentence = answer_item["text"].strip().lower()
    return paraphrased_sentence


the_seed_sentence = "a 46-year-old relatively handsome man"
the_adj_words = get_adjective_words(the_seed_sentence)
print("the_adj_words:", the_adj_words)
the_analogy_list = []
for i in range(len(the_adj_words)):
    adj_word_item = the_adj_words[i]
    p_adj_w, n_adj_w = adj_word_item[0].lower(), adj_word_item[1].lower()
    
    analogy = get_analogy(the_seed_sentence, n_adj_w)
    the_analogy_list.append(analogy)    

print("the_analogy_list:", the_analogy_list)
the_analogy = the_analogy_list[0]
the_paraphrased_sentence_list = []
for i in range(len(the_adj_words)):
    adj_word_item = the_adj_words[i]
    p_adj_w, n_adj_w = adj_word_item[0].lower(), adj_word_item[1].lower()

    analogy = the_analogy_list[i]

    paraphrased_sentence = paraphrase_by_attractiveness("man", 46, p_adj_w, n_adj_w, analogy)
    # print("paraphrased_sentence:", paraphrased_sentence)
    the_paraphrased_sentence_list.append(paraphrased_sentence)


# the_analogy = get_analogy("a 46-year-old relatively handsome man", "weak")
# print("the_analogy:", the_analogy)



# image_url = "https://assets.deutschlandfunk.de/98e549f6-ae2b-4795-967b-2ae72e16a94a/1920x1080.jpg?t=1682401566674"
# the_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
# (contains_face, cropped_image) = get_cropped_face_image(the_image)
# if (contains_face):
#     print("Face detected")
#     # cropped_image.show()
#     face_emotion_and_properties = get_face_emotion_and_other_properties(the_image)
#     print("face_emotion_and_properties:", face_emotion_and_properties)

#     (attractiveness_index, attractiveness_score) = get_attractiveness_of_face(cropped_image, face_emotion_and_properties["dominant_gender"])
#     print("attractiveness_index:", attractiveness_index)
#     print("attractiveness_score:", attractiveness_score)

#     sacarstic_seed = generate_sacarstic_seed_by_attractiveness(attractiveness_index, attractiveness_score, face_emotion_and_properties["dominant_gender"], face_emotion_and_properties["age"])
#     print("sacarstic_seed:", sacarstic_seed)

