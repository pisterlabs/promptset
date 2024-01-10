from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from Family_Album_INTEG.models.face_recognition import face_recognition
from Family_Album_INTEG.models.rembg.rembg.bg import remove
from Family_Album_INTEG.models.LLaVA.llava.serve import cli
from Family_Album_INTEG.models.LLaVA.llava.serve.cli import load_custom_model,inference_image
import numpy as np
from geopy.geocoders import Nominatim

from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI

################# gpt4v output format ##################
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
import re
import json

# class Album_Output(BaseModel):
#     search_keywords : List[str] = Field(description="Backgrounds and objects that appear in photos.")
#     summary: str = Field(description="One sentence to summarize the photo.")
#     emoji: str = Field(description="One emoji to go with a sentence summarizing the photo.")
#     emoji_unicode: str = Field(description="Unicode-encoded value of one emoji that matches the sentence summarizing the photo.")


class Album_Output(BaseModel):
    search_keywords : List[str] = Field(description="Backgrounds and objects that appear in photos.")
    summary: str = Field(description="One sentence to summarize the photo.")
################# gpt4v output format ##################

class Image_Processor():

    def __init__(self):
        pass

    # 촬영 날짜/시간, 위도/경도 추출
    def get_metadata(self,img):

        img = Image.open(img)

        date_time = ''
        latitude,longitude = 0,0
        location=''

        try:
            # with Image.open(image_path) as img:
            # 이미지의 Exif 데이터 읽기
            exif_data = img._getexif()

            # print(exif_data)
            if exif_data:
                date_time = None
                gps_info = None

                for tag, value in exif_data.items():
                    tag_name = TAGS.get(tag, tag)

                    # 촬영 날짜 및 시간 추출 (DateTimeOriginal 또는 DateTime 태그 사용)
                    if tag_name == 'DateTimeOriginal' or tag_name == 'DateTime':
                        date_time = value

                    # GPS 정보 추출
                    if tag_name == 'GPSInfo':
                        gps_info = {}
                        for gps_tag, gps_value in value.items():
                            gps_tag_name = GPSTAGS.get(gps_tag, gps_tag)
                            gps_info[gps_tag_name] = gps_value

                # 촬영 날짜 및 시간 출력
                if date_time:
                    print(f"촬영 날짜 및 시간: {date_time}")
                    pass
                # GPS 정보 출력
                if gps_info:
                    latitude = gps_info.get('GPSLatitude', None)
                    longitude = gps_info.get('GPSLongitude', None)
                    # print("latitude : ",latitude)
                    # print("longitude : ",longitude)
                    if latitude and longitude:
                        print("#### get_metadata ####")
                        print("latitude:",latitude)
                        print("longitude:",longitude)
                        print("#### get_metadata ####")

                        if 'nan' in str(latitude):
                            print("lat : ",latitude)
                            print("lon : ",longitude)
                            latitude = 0
                            longitude = 0
                        # lat = f"{latitude[0]}° {latitude[1]}' {latitude[2]}'' {gps_info['GPSLatitudeRef']}"
                        # lon = f"{longitude[0]}° {longitude[1]}' {longitude[2]}'' {gps_info['GPSLongitudeRef']}"
                        # lat = f"{latitude[0]}° {latitude[1]}' {latitude[2]}'' {gps_info['GPSLatitudeRef']}"
                        geolocator = Nominatim(user_agent="coordinateconverter")
                        latitude = round(float(latitude[0])+float(latitude[1])/60+float(latitude[2])/3600,6)
                        # lon = f"{longitude[0]}° {longitude[1]}' {longitude[2]}'' {gps_info['GPSLongitudeRef']}"
                        longitude = round(float(longitude[0])+float(longitude[1])/60+float(longitude[2])/3600,6)

                        print("latitude:",latitude)
                        print("longitude:",longitude)

                        ###############################
                        ### 11_22 address in Korean ###
                        ###############################
                        # location = geolocator.reverse(str(latitude)+", "+str(longitude)) #before
                        location = geolocator.reverse(str(latitude)+", "+str(longitude),language='ko') # after

                        location = location.address
                        print(f"촬영 위치 (GPS): 위도 {latitude}, 경도 {longitude}")

                        #############################
                        ##### 12_4 nan latitude #####
                        #############################
                        # if 'nan' in str(latitude):
                        #     print("lat : ",latitude)
                        #     print("lon : ",longitude)
                        #     latitude = 0
                        #     longitude = 0
                    else:
                        print("촬영 위치 (GPS) 정보 없음")
                        pass
            else:
                print("이미지에 Exif 메타데이터가 없습니다.")

        except Exception as e:
            print(f"메타데이터를 추출하는 동안 오류가 발생했습니다: {e}")

        return date_time,latitude,longitude,location
    
    def extract_character(self,img):
        height, width, channels = 640, 640, 3
        black_image = Image.fromarray(np.zeros((height, width, channels), dtype=np.uint8))
        # black_image = Image.open(black_image)

        # 인물 이미지 (PIL Image로 읽기 및 크기 조정)
        person_image = Image.open(img)

        person_image = person_image.resize((640, 640))

        # 인물 마스킹 이미지 (PIL Image로 읽기 및 크기 조정)
        mask_image = remove(person_image,only_mask=True).convert("L") 
        mask_image = mask_image.point(lambda p: p > 128 and 255)  # 0 또는 255로 수정
        mask_image = mask_image.resize((640, 640))

        # 배경 이미지와 마스킹된 인물 이미지를 결합하여 새로운 이미지 생성 (PIL Image로)
        result = Image.composite(person_image, black_image, mask_image)

        return result


    def face_recognition(self,img,face_encoding_dict):
    # def extract_member(img,member_id,member_encoding):

        # print("face_registration_db.values() : ", np.array(face_registration_db.values()))
        # print("face_registration_db.keys() : ",face_registration_db.keys())

        characters = []
        nicknames = []
        encodings = []

        for id,encoding in face_encoding_dict.items():
            nicknames.append(id)
            encodings.append(encoding)

        unknown_image = face_recognition.load_image_file(img)
        face_locations = face_recognition.face_locations(unknown_image)
        # print("face_locations : ",face_locations)
        face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(encodings, face_encoding)

            name = "Unknown"

            face_distances = face_recognition.face_distance(encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            # print("best_match_index : ",best_match_index)

            ##############################################
            # face recoginiton probabliity 
            # print("face_distances : ",face_distances)

            if matches[best_match_index]:
                name = nicknames[best_match_index]
            characters.append(name)

        return characters
    
    #####################
    ##### with llava ####
    #####################
    def load_llava_model(self):

        tokenizer, model, image_processor, context_len,image_aspect_ratio,roles,conv,temperature,max_new_tokens,debug = load_custom_model()

        return tokenizer, model, image_processor, context_len,image_aspect_ratio,roles,conv,temperature,max_new_tokens,debug
    
    def llava_inference_image(self,
                    tokenizer,
                    model,
                    image_processor,
                    context_len,
                    image_aspect_ratio,
                    roles,
                    conv,
                    temperature,
                    max_new_tokens,
                    debug,
                    image_file,
                    character_origin):
        
        inference_outputs = inference_image(tokenizer,
                    model,
                    image_processor,
                    context_len,
                    image_aspect_ratio,
                    roles,
                    conv,
                    temperature,
                    max_new_tokens,
                    debug,
                    image_file,
                    character_origin)
        
        return inference_outputs
    
    #####################
    ##### with gpt4v ####
    #####################

    def load_gpt4v_model(self):

        # Instantiate the parser with the new model.
        parser = PydanticOutputParser(pydantic_object=Album_Output)

        # Update the prompt to match the new query and desired format.
        prompt = ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template(
                    "answer the users question as best as possible.\n{format_instructions}\n{question}"
                )
            ],
            input_variables=["question"],
            partial_variables={
                "format_instructions": parser.get_format_instructions(),
            },
        )

        # 1. secret
        # 2. secret_aivil

        with open('resource/secret_minkyo.json', 'r') as json_file:
            data = json.load(json_file)
        openai_api_key = data['openai_api_key']

        chat_model = ChatOpenAI(
            model="gpt-4-vision-preview",
            openai_api_key=openai_api_key,
            max_tokens=1024,
            temperature=1 # 12/9 0.7 
        )

        return parser, prompt, chat_model
    

    def get_user_query(self,img_url,characters,photo_datetime,photo_location):

        # Generate the input using the updated prompt.
        # user_query = f"""
        #             너는 인공지능을 활용한 사진 앨범 기능을 도와주는 역할이야. 앨범에는 주로 가족들, 지인들과 찍은 사진이거나 개인 일상 사진들이 저장돼.
        #             예를 들어, 가족들과 함께 프랑스 파리로 여행간 사진, 부부끼리 주말에 레스토랑에 외식하러간 사진, 혼자 카페에서 브런치 먹는 사진 등 이야.

        #             입력된 이미지는 {img_url}이야. 이미지를 참고하여 너의 임무를 수행해줘.너에게 맞겨진 임무는 4가지야.

        #             첫번째로, 입력된 사진에서 나타나는 배경,장소,날씨,물체,의상,건물,국가,도시 등 특징들을 단어로 나열하는 것이야.
        #             이 첫번째 임무는 사용자가 앨범에서 사진을 찾을 때, 생각한 이미지의 특징들을 기반으로 검색 키워드(단어)를 입력하여 유연한 검색이 가능하게끔 하기 위함이야.
        #             반드시, 사진에 나타나는 대상들을 10~15개의 단어로 나열해서 추출해줘. 예를 들어, '커플이 화창한 날 모래 해변 앞에서 찍은 사진'이 입력된다면 
        #             ['바다','해변','파도','물결','등대','모래','맑음','휴가','여행','바캉스','청바지','선글라스','반팔티'] 와 같이 python의 리스트 형식으로 말해줘.

        #             두번째로, 입력된 사진을 기반으로 한 문장으로 요약하는 것이야. 입력된 사진, 첫번째 임무에서 추출된 단어들, 추가적으로 입력해줄 다음과 정보들을 기반으로 한 문장을 생성하면 돼.
        #             사진의 등장인물들은 {characters} 이고, 촬영날짜는 {photo_datetime}이고, 촬영위치는 {photo_location}이야. 이러한 정보들을 토대로 '행복한', '즐거운', '소중한'
        #             ,'추억' 등의 감성적인 단어들을 다양하게 조합하여 말해주면 돼. 또한 반드시 등장인물이나 배경(해변, 산, 공원, 카페, 레스토랑, 영화관, 백화점, 프랑스 파리, 부산 해운대 등)을 중심으로 서술해줘.
        #             예시를 들자면, '크리스마스 날, 아늑한 카페에서 헤리랑 정이랑~', '사랑하는 정이와 혜리 부산 바캉스에서', '프랑스 파리 에펠탑에서 낭만있게 한 장!','새해 맞이 등산길 정상에서 정이!','고급 레스토랑에서 분위기있게 외식하며',
        #             '사랑하는 혜리와 일본 도톤보리에서', '혜리랑 백화점에서 신나게 쇼핑할 때~!' 처럼 말해줘. 지인들끼리 공유하는 앨범이기 때문에 최대한 귀엽고 재밌게 요약해줘. 요약할 문장을 길이는 반드시 텍스트로만 15~18글자로 작성해줘. 

        #             세번째로, 두번째 임무를 통해 요약한 문장과 어울리는 이모티콘 1개를 말해줘. 이모티콘은 유니코드로 변환 가능한 이모티콘으로 출력해줘. 반드시 문장과 어울리는 깜찍하고 귀여운 이모티콘을 제시해야하고 갯수는 1개야.
                    
        #             네번쨰로, 세번째 임무를 통해 얻은 이모티콘을 유니코드로 인코딩해서 알려줘. 반드시 이모티콘과 맵핑된 유니코드로 알려줘야해. 

        #             첫번째,두번째,세번째,네번째 임무의 결과값은 반드시 JSON 파일 형식으로 반환해줘. key는 "serach_keywords", "summary", "emoji", "emoji_unicode"야. 반드시 이해했다는 답변은 하지 말고 JSON 파일 형식의 값만 반환해줘.

        #         """


        user_query = f"""
            너는 인공지능을 활용한 사진 앨범 기능을 도와주는 역할이야. 앨범에는 주로 가족들, 지인들과 찍은 사진이거나 개인 일상 사진들이 저장돼.
            예를 들어, 지인들과 함께 프랑스 파리로 여행간 사진, 시원한 바다 앞에서 찍은 기념 사진, 아늑한 카페에서 커피를 즐기는 사진 등 이야.

            입력된 이미지는 {img_url}이야. 이미지를 참고하여 너의 임무를 수행해줘.너에게 맞겨진 임무는 4가지야.

            첫번째로, 입력된 사진에서 나타나는 배경,장소,날씨,물체,의상,건물,국가,도시 등 특징들을 단어로 나열하는 것이야.
            이 첫번째 임무는 사용자가 앨범에서 사진을 찾을 때, 생각한 이미지의 특징들을 기반으로 검색 키워드(단어)를 입력하여 유연한 검색이 가능하게끔 하기 위함이야.
            반드시, 사진에서 시각적으로 보이는 배경이나 물체들을 10~15개의 단어로 나열해서 추출해줘. 예를 들어, '커플이 화창한 날 모래 해변 앞에서 찍은 사진'이 입력된다면 
            ['바다','해변','파도','물결','등대','모래','맑음','휴가','여행','바캉스','청바지','선글라스','반팔티'] 와 같이 말해주거나
            '아늑한 카페에서 찍은 사진'이 입력되다면 ['카페','커피','케이크','디저트','티스푼','샌드위치','테이블','의자','메뉴판'] 와 같이 말해주거나
            '아쿠아리움 앞에서 찍은 기념 사진'이 입력된다면 ['아쿠아리움','수족관','물고기'.'상어','물결','수중터널','동물','바다'] 와 같이 python의 리스트 형식으로 말해줘.
            이때, 반드시 한글로 이루어진 단어들로만 나열해줘.

            두번째로, 입력된 사진을 기반으로 한 문장으로 요약하는 것이야. 입력된 사진, 첫번째 임무에서 추출된 단어들, 추가적으로 입력해줄 다음과 정보들을 기반으로 한 문장을 생성하면 돼.
            사진의 등장인물들은 {characters} 이고, 촬영날짜는 {photo_datetime}이고, 촬영위치는 {photo_location}이야. 이러한 정보들을 토대로 '행복한', '즐거운', '사랑스러운','소중한'
            ,'추억' 등의 감성적인 단어들을 다양하게 조합하여 말해주면 돼. 또한 반드시 등장인물이나 배경(해변, 산, 공원, 카페, 레스토랑, 영화관, 백화점 등)을 중심으로 서술해줘.
            예시를 들자면, '크리스마스 날, 호텔 아늑한 카페에서 헤리랑 정이랑~', '사랑하는 정이와 혜리 부산 바캉스에서', '프랑스 파리 에펠탑 앞에서 낭만있게 한 장!','새해 맞이 등산길 정상에서 정이!', '사랑하는 혜리와 일본 도톤보리에서'처럼 말해줘.
            지인들끼리 공유하는 앨범이기 때문에 최대한 귀엽고 재밌게 요약해줘. 요약할 문장을 길이는 반드시 텍스트로만 15~18글자로 작성해줘. 

            첫번째,두번째 임무의 결과값은 반드시 JSON 파일 형식으로 반환해줘. key는 "serach_keywords", "summary"야. 반드시 이해했다는 답변은 하지 말고 JSON 파일 형식의 값만 반환해줘.

        """
        user_query = user_query.format(img_url=img_url,characters=characters,photo_datetime=photo_datetime,photo_location=photo_location)

        # print("get_user_query : ",user_query)
        print("photo_location : ",photo_location)
        return user_query