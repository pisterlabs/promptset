import json
import textwrap
from typing import List, Tuple, Union
from io import BytesIO

import cloudinary
import cloudinary.uploader
import numpy as np
import openai
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline

from config import settings


openai.api_key = settings.api_key

cloudinary.config(
    cloud_name=settings.cloud_name,
    api_key=settings.cloud_api_key,
    api_secret=settings.cloud_api_secret,
)

class Service:
    def __init__(self) -> None:
        self.ocr_model = PaddleOCR(
            rec_model_dir=settings.ocr_model_path,
            config=f"{settings.ocr_model_path}/config.yml",
            rec_char_dict_path=f"{settings.ocr_model_path}/latin_dict.txt",
            use_angle_cls=True,
            use_space_char=True,
            lang="latin",
        )
        self.en_to_fr = pipeline(
            "translation",
            model=settings.en_to_fr_model_path, 
        )
        self.fr_to_en = pipeline(
            "translation",
            model=settings.fr_to_en_model_path, 
        )

        self.prompt_en = "Correct only the grammar and the orthograph of the following text:\n\n {sentence}"
        self.prompt_fr = "Corrige seulement la grammaire et l'orthographe du texte suivant :\n\n {sentence}"

    def inference(self, image:Image.Image, source_lang: str, target_lang: str) -> Tuple[Union[Image.Image, str], str, str, str]:
        """
        Inference lance le processus de traduction du texte sur une image.

        Args:
            image (Image.Image): Image à traiter
            source_lang (str): Langue source (avant traduction)
            target_lang (str): Langue cible (après traduction)
        
        Returns:
            Image.Image: Image avec le texte traduit

        """
        ocr_results = self.do_ocr(image)
        ocr_texts = [line[1][0] for line in ocr_results]
        concatenated_ocr_texts = " ".join(ocr_texts)
        
        # ocr_confidence = [line[1][1] for line in ocr_results]
        input_sentence = concatenated_ocr_texts

        if input_sentence == "":
        
            return None

        else:

            try:
                corrected_sentences = self.do_correct_text(input_sentence, source_lang)
                print("===============")
                print(corrected_sentences)
                print(len(corrected_sentences))
                print("===============")
            except openai.error.RateLimitError as e:
                print(e)
                corrected_sentences = input_sentence
                
            print(type(corrected_sentences))
            corrected_sentences_splitted = corrected_sentences.split("\n")

            with open("corrected_sentences.json", "w") as f:
                json.dump(corrected_sentences, f, indent=4)

            # ocr_bbox = [line[0] for line in ocr_results]
            # translated_texts = self.do_translation(ocr_texts, source_lang, target_lang)
            translated_paragraph = self.do_translation(corrected_sentences_splitted, source_lang, target_lang)
            print("****translated_paragraph****")
            print(translated_paragraph)
            print("****translated_paragraph****")
            
            post_processed_image = self.do_post_processing_for_paragraphs(translated_paragraph)

        return post_processed_image, concatenated_ocr_texts, corrected_sentences, translated_paragraph
        
    def do_ocr(self, image:Image.Image) -> List[List[Union[List[List[float]], Tuple[str, float]]]]:
        """
        Do OCR lance le modèle d'OCR sur une image.

        Args: 
            image (Image.Image): Image à traiter

        Returns:
            List[List[Union[List[List[float]], Tuple[str, float]]]]: Liste de liste de listes ([bbox], (texte, confiance))
        """
        pix = np.array(image.convert('RGB'))
        result_base = self.ocr_model.ocr(pix)
        
        print(result_base[0])
        return result_base[0]
    
    def do_correct_text(self, sentence:str, source_lang: str):
        """
        Do correct text lance le modèle de correction orthographique et grammaticale sur le texte.

        Args:
            sentence (str): Phrase à corriger
            source_lang (str): Langue source (avant traduction)

        Returns:
            str: Phrase corrigée
        """
        if source_lang == "en":
            prompt= self.prompt_en.format(sentence=sentence)
        if source_lang == "fr":
            prompt= self.prompt_fr.format(sentence=sentence)    
         
        completion = openai.Completion.create(  
        model="text-davinci-003", 
        prompt= prompt,
        temperature=0.3,
        max_tokens=2048,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
        )

        return completion['choices'][0].text
    

    def do_translation(self, ocr_texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        """
        Do translation lance le modèle de traduction sur la liste de textes sortie par l'OCR.

        Args:
            ocr_texts (List(str)): Liste de textes à traduire
            source_lang (str): Langue source (avant traduction)
            target_lang (str): Langue cible (après traduction)
        
        Returns:
            List(str): Liste de textes traduits

        Raises:
            ValueError: Si la langue source n'est pas 'en' ou 'fr'
        """

        liste_filtree = [element for element in ocr_texts if len(element)>10 ]
        print("**************liste_filtree*************")
        print(liste_filtree)
        print("**************liste_filtree*************")

        if source_lang == "en":
            translated_texts = self.en_to_fr(liste_filtree)
        elif source_lang == "fr":
            translated_texts = self.fr_to_en(liste_filtree)
        else:
            raise ValueError("Source language must be either 'en' or 'fr'")
        
        return [line["translation_text"] for line in translated_texts]
    
    
    def do_post_processing_for_paragraphs(self, translated_paragraphs: list()) -> Image.Image:
        """
        Do post processing paragraphs applique un rectangle opaque sur une image de taille 600*600 et écrit les paragraphes traduits sur l'image.
        Chaque string de la liste translated_paragraphs correspond à une ligne de texte.
        Si le texte est trop long, il est coupé en plusieurs lignes grâce à la fonction textwrap.wrap.

        Args:
            image (Image.Image): Image de base
            translated_paragraphs (List[str]): Liste de paragraphes traduits
        
        Returns:
            Image.Image: Image de base avec le rectangle et les paragraphes traduits
        """
          
        # Créer une image de fond
        width, height = 600, 600
        background_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))

        # Créer un nouvel objet ImageDraw pour dessiner sur l'image de fond
        draw = ImageDraw.Draw(background_image)

        # Définir la police et la taille du texte
        font = ImageFont.truetype("arial.ttf", 14)

        # Vérifier si la liste contient une seule chaîne de caractères
        if len(translated_paragraphs) == 1 and isinstance(translated_paragraphs[0], str):
            lines = textwrap.wrap(translated_paragraphs[0], width=70)
            y = 75  # Position du texte en haut de l'image
            for line in lines:
                text_width, text_height = draw.textsize(line, font=font)
                x = (width - text_width) / 2
                draw.text((x, y), line, font=font, fill="black")
                y += text_height + 10  # Espacement entre les lignes

        elif len(translated_paragraphs) > 1:
            y = 75  # Position du texte en haut de l'image
            for text in translated_paragraphs:
                if isinstance(text, str):
                    lines = textwrap.wrap(text, width=70)
                    for line in lines:
                        text_width, text_height = draw.textsize(line, font=font)
                        x = (width - text_width) / 2
                        draw.text((x, y), line, font=font, fill="black")
                        y += text_height + 10  # Espacement entre les lignes

        # Charger l'image à coller
        image_to_paste = Image.open("dishdecoderlogo2.png").convert("RGBA")

        # Redimensionner l'image à la taille de l'image de fond
        image_to_paste.thumbnail((width, int(height/4)))

        # Coller l'image sur l'image de fond
        background_image.paste(image_to_paste, (70, 0), mask=image_to_paste)

        return background_image
    
    def image_to_cloud(self, image: Image.Image) -> str:
        """
        Image to cloud envoie l'image sur Cloudinary et retourne l'url de l'image.
        Args:
            image (Image.Image): Image à envoyer sur le cloud
        
        Returns:
            str: Url de l'image sur le cloud
        """

        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes.seek(0)

        response = cloudinary.uploader.upload(image_bytes)
        image_url = response["secure_url"]

        return image_url
   
    
        
    
