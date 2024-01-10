import tensorflow as tf
import openai
import os

from transformers import AutoImageProcessor, AutoTokenizer, TFVisionEncoderDecoderModel

class Predict:

    def __init__(self):

        with open('api.txt') as f:
            api = f.readlines()

        openai.api_key = api[0]

        encoder_pretrained = 'google/vit-base-patch16-224'
        decoder_pretrained = 'indolem/indobert-base-uncased'

        self.image_processor = AutoImageProcessor.from_pretrained(encoder_pretrained)
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_pretrained)

        self.model_category_dict = {
            "aksesoris_fashion": "vit_bert_aksesoris_fashion",
            "buku_alat_tulis": "vit_bert_buku_alat_tulis",
            "elektronik": "vit_bert_elektronik",
            "fashion_bayi_anak": "vit_bert_fashion_bayi_anak",
            "fashion_muslim": "vit_bert_fashion_muslim",
            "fotografi": "vit_bert_fotografi",
            "handphone_aksesoris": "vit_bert_handphone_aksesoris",
            "hobi_koleksi": "vit_bert_hobi_koleksi",
            "ibu_bayi": "vit_bert_ibu_bayi",
            "jam_tangan": "vit_bert_jam_tangan",
            "kesehatan": "vit_bert_kesehatan",
            "komputer_aksesoris": "vit_bert_komputer_aksesoris",
            "makanan_minuman": "vit_bert_makanan_minuman",
            "olahraga_outdoor": "vit_bert_olahraga_outdoor",
            "otomotif": "vit_bert_otomotif",
            "pakaian_pria": "vit_bert_pakaian_pria",
            "pakaian_wanita": "vit_bert_pakaian_Wanita",
            "perawatan_kecantikan": "vit_bert_perawatan_kecantikan",
            "perlengkapan_rumah": "vit_bert_perlengkapan_rumah",
            "sepatu_pria": "vit_bert_sepatu_pria",
            "sepatu_wanita": "vit_bert_sepatu_wanita",
            "souvenir_party_supplies": "vit_bert_souvenir_party_supplies",
            "tas_pria": "vit_bert_tas_pria",
            "tas_wanita": "vit_bert_tas_wanita",
        }


    def predict(self, pillow_image, category):
        
        if len(tf.config.list_physical_devices('GPU')) > 0:
            print("Use GPU")
            print("Loading Model")
            model_path = f'models/{self.model_category_dict[category]}'
            model = TFVisionEncoderDecoderModel.from_pretrained(model_path)

            print("Preprocess Image")
            pixel_values = self.image_processor(pillow_image, return_tensors="tf").pixel_values

            print("Generate Text")
            generated_ids = model.generate(pixel_values)

            print("Decode Text")
            generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            with tf.device('/cpu:0'):
                print("Use CPU")
                print("Loading Model")
                model_path = f'models/{self.model_category_dict[category]}'
                model = TFVisionEncoderDecoderModel.from_pretrained(model_path)

                print("Preprocess Image")
                pixel_values = self.image_processor(pillow_image, return_tensors="tf").pixel_values

                print("Generate Text")
                generated_ids = model.generate(pixel_values)

                print("Decode Text")
                generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_text


    def predict_improvement(self, generated_title):

        title_prompt = f'Fix this product title and make it compelling and eye catching "{generated_title}". Only reply with the title in Bahasa Indonesia, do not add any explanation'
        predicted_title = self.get_result(title_prompt).replace('"', '')

        description_prompt = f'Write a description for product titled "{predicted_title}", make it three sentences long. Only answer with the description in Bahasa Indonesia'
        predicted_description = self.get_result(description_prompt)

        return predicted_title, predicted_description
    
     
    def get_result(self, prompt):
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        content = completion.choices[0].message.content

        return content
