import os
import time
import json
import shutil
import pandas as pd
from glob import glob
from datetime import datetime
import torch
import openai
import requests
import uuid
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from paddleocr import PaddleOCR, draw_ocr
from ultralytics import YOLO


class ImageProcess:
    def __init__(self, openai_key, clova_api_url, clova_secret_key):
        self.model = YOLO('best.pt')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.openai_key = openai_key
        self.clova_api_url = clova_api_url
        self.clova_secret_key = clova_secret_key
        self.class_names = ['banner','frame']
        self.predict_path = "runs/detect/predict/"
        self.predict_detect_path = self.predict_path
        self.predict_crop_banner_path = self.predict_path + self.class_names[0]
        self.predict_crop_frame_path = self.predict_path + self.class_names[1]
        self.log_path = 'logs/'
        self.ocr = PaddleOCR(lang = 'korean')
        # 추가 ////////
        self.date_created = ''
        self.imgs = glob('./check_flask/views/capture_data/*.jpg')
        self.json_file_path = './check_flask/views/capture_data/meta_data.json'
        openai.api_key = self.openai_key

    def make_frame(self):
        # encoding 추가 ////////
        with open(self.json_file_path, 'r', encoding='UTF-8') as json_file:
            meta_data = json.load(json_file)

        # DataFrame 정의
        df_data = []
        for item in tqdm(meta_data['data']):
            date_time = item['timestamp'].split("T")
            df_row = {
                'ID' : item['id'],
                'Date' : date_time[0],
                'Time' : date_time[1],
                'Location' : [item['location']['latitude'], item['location']['longitude']],
                'Origin_img' : [np.array(Image.open('./check_flask/views/capture_data/' + item['file_name'])).tolist()],
                'Detect_img' : [],
                'Crop_classes' : [],
                'Crop_imgs' : [],
                'Crop_xyxy' : [], 
                'Crop_conf': [],
                'PaddleOCR_text' : [],
                'ClovaOCR_text' : [],
                'Category' : [],
                'Category_basis' : [],
                'Legality' : []
            }
            df_data.append(df_row)

        df_report = pd.DataFrame(df_data)
        self.date_created = meta_data['dataset_info']['date_created'].split("T")[0].split("-")
        time.sleep(0.1)
        # print(date_created)
        df_report.to_csv('./check_flask/views/reports/report_' + '_'.join(self.date_created) + '.csv')
        return df_report
    
    def move_all_img(self, source_folder, destination_folder):
        # source_folder 내의 모든 항목을 destination_folder로 이동
        for item in os.listdir(source_folder):
            source_item = os.path.join(source_folder, item)
            destination_item = os.path.join(destination_folder, item)
            shutil.move(source_item, destination_item)

    def move_img(self, source_path, destination_path):
        if not os.path.exists(os.path.dirname('/'.join(destination_path.split('/')[:-1])+'/')):
            os.makedirs(os.path.dirname('/'.join(destination_path.split('/')[:-1])+'/'), exist_ok=True)        
        # 파일 이동
        shutil.move(source_path, destination_path)


    def check_category(self, df_report,id,image,crop_classes,crop_xyxy):
        # resize된 상태에서 xy좌표를 뽑기 때문에 다시 resize
        image = np.resize(image,(640,640))
        # print("원본 사이즈에서 리사이즈 : ",np.resize(image,(640,640)).shape)
        frameXYXY_list = []
        for idx,class_name in enumerate(crop_classes):
            if class_name=='frame':
                frameXYXY_list.append(crop_xyxy[idx])
        
        for i,frameXYXY in enumerate(frameXYXY_list):
            print(str(i+1)+"번째 frameXYXY :",frameXYXY)
            for j,cropXYXY in enumerate(crop_xyxy):
                if crop_classes[j]!='frame':
                    print(str(i+1)+"번째 cropXYXY :",cropXYXY)

                    frame_min_x, frame_min_y, frame_max_x, frame_max_y = frameXYXY
                    crop_min_x, crop_min_y, crop_max_x, crop_max_y = cropXYXY

                    tmp_origin_img = np.array(image.copy(),dtype=np.int16)
                    tmp_origin_img[frame_min_x:frame_max_x,frame_min_y:frame_max_y] = -1

                    target_region = tmp_origin_img[crop_min_x: crop_max_x, crop_min_y :crop_max_y]
                    # print("target_region.size : ",target_region.size)
                    negative_one_pixel_count = np.count_nonzero(target_region == -1)
                    total_pixel_count = target_region.size
                    negative_one_percentage = (negative_one_pixel_count / total_pixel_count) * 100

                    print(str(i+1)+"번째 frame"+str(j+1)+"번째 crop이미지 겹치는 범위 : ",negative_one_percentage)

                    if negative_one_percentage >= 70:
                        df_report.iloc[id]['Category'][j] = 1
                        df_report.iloc[id]['Category_basis'][j] = 1
        return df_report

    #TODO: Object Detecting by YOLO 
    def yolo_run(self, img, df_report):
        results = self.model.predict(
                                source=img, # 디렉토리 (capture_data/)
                                conf=0.5, # confidence threshold for detection (오탐지 시 재설정)
                                save=True,  # Detect 결과 저장 (runs/detect/predict)
                                device=self.device, # device 설정
                                show=False, # window 창으로 show
                                save_crop=True # Detect된 Obeject 사진 저장 (runs/detect/predict/crops)
                                )
        
        image = np.array(Image.open(img))

        for idx,result in enumerate(results):
            now = datetime.now()
            now_time = str(now.year) + str(now.month) + str(now.day) + '_' + str(now.hour) + str(now.minute) + str(now.second)

            boxes = result.boxes
            saved_img = ''

            print("================================= Predict 결과 =================================")

            file_name = img.split('\\')[-1]
            img_name = file_name[:-4]
            # ext_name = file_name[-4:]

            ### 데이터 logs에 저장
            # 데이터이름+날짜시간
            data_datetime_dir = self.log_path+now_time+'_'+img_name+'/'

            # 오류나서 추가1
            os.makedirs(data_datetime_dir)

            # predict된 결과를 'logs/'에 저장
            self.move_all_img(self.predict_path,data_datetime_dir)

            # 오류나서 추가2
            time.sleep(1)
            # 원본 이미지 저장
            os.makedirs(data_datetime_dir+'origin_img/')
            shutil.copyfile(img,data_datetime_dir+'origin_img/'+img.split('\\')[-1])
            # detect된 이미지 저장
            self.move_img(data_datetime_dir+file_name,data_datetime_dir+'detect_img/'+file_name)
            
            try:
                # crop_imgs로 폴더명 변경
                os.rename(data_datetime_dir+'crops', data_datetime_dir+'crop_imgs')
            except:
                print("Crop된 이미지 없음! (Detect이미지 없음!)")

            ### 데이터 df에 저장
            id = int(img_name.split('_')[-1])-1
            # detect_img 기록
            df_report['Detect_img'].iloc[id] = np.array(Image.open(data_datetime_dir+'detect_img/'+file_name)).tolist()

            crop_classes = []
            crop_imgs = []
            crop_xyxy = []
            crop_conf = []
            for idx,box in enumerate(boxes):
                # crop된 object 이름 기록
                crop_classes.append(self.class_names[int(box.cls)])
                # crop된 xyxy 좌표 기록
                xyxy = list(box.xyxy[0].to('cpu').numpy().astype('int'))
                x_min, y_min, x_max, y_max = xyxy
                crop_xyxy.append([x_min, y_min, x_max, y_max])
                # crop된 이미지 기록
                crop_imgs.append(image[y_min:y_max, x_min:x_max,:].tolist())
                # crop된 이미지별 conf 기록
                crop_conf.append(box.conf.detach().cpu().numpy().astype('float32')[0])

            # crop된 object 이름 기록
            df_report['Crop_classes'].iloc[id] = crop_classes
            df_report['Crop_imgs'].iloc[id] = crop_imgs
            df_report['Crop_xyxy'].iloc[id] = crop_xyxy
            df_report['Crop_conf'].iloc[id] = crop_conf

            df_report['Category'].iloc[id] = [0 if class_name=='frame' else -1 for class_name in crop_classes]
            df_report['Category_basis'].iloc[id] = [0 if class_name=='frame' else -1 for class_name in crop_classes]
            df_report = self.check_category(df_report,id,image,crop_classes,crop_xyxy)
            df_report.to_csv('check_report')
        return df_report
    
# ----------------여기까지 진행함 ----------------------
    # TODO: Recognition Texts with CLOVA OCR
    def clova_ocr(self, img_path):
        request_json = {
            'images': [
                {
                    'format': 'jpg',
                    'name': 'demo'
                }
            ],
            'requestId': str(uuid.uuid4()),
            'version': 'V2',
            'timestamp': int(round(time.time() * 1000))
        }

        payload = {'message': json.dumps(request_json).encode('UTF-8')}
        files = [
        ('file', open(img_path,'rb'))
        ]
        # secret_key -> clova_secret_key 변경 ////////
        headers = {
        'X-OCR-SECRET': self.clova_secret_key
        }
        # api_url -> clova_api_url로 변경 ////////
        response = requests.request("POST", self.clova_api_url, headers=headers, data = payload, files = files)
        return response

    def get_clova_contents(self, img_path):        
        response = self.clova_ocr(img_path)
        response = response.json()
        contents = []
        for field in response['images'][0]['fields']:
            text = field['inferText']
            contents.append(text)
        return contents
    
    def clova_ocr_run(self, idx, df_report):
        n_crops = len(df_report.iloc[idx]['Crop_classes'])
        
        for i in range(n_crops):
            # 만약 'banner'면 해당 사진의 contents를 추출해 저장
            if df_report.loc[idx]['Crop_classes'][i] == 'banner':
                img_path = 'naver_ocr_temp.jpg'
                image = Image.fromarray(np.uint8(df_report.loc[idx]['Crop_imgs'][i]))
                image.save(img_path, 'jpeg')
                contents = self.get_clova_contents(img_path)
                df_report.loc[idx,'ClovaOCR_text'].append(contents)
            # frame 이면 빈 list 저장 (인덱스 맞춰주어야함)
            else:
                df_report.loc[idx,'ClovaOCR_text'].append(['frame'])
        return df_report

    ## TODO: TO CLASSIFY with GPT
    def classify_text(self, text):
        text = ' '.join(text)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[
                {"role": "system", "content": "You are responsible for classifying the text of advertising banners near the road or on the street."},
                {"role": "system", "content": "There are a total of three classes of advertising banners to classify."},
                {"role": "system", "content": "The Class 1 is the text of the public service banner installed by the city hall and district office."},
                {"role": "system", "content": "The Class 2 is  the text of a political promotion banner set up by politicians."},
                {"role": "system", "content": "The Class 3 is all banners other than 1 and 2. For example, text such as a hospital, gym, or academy promotional banner."},
                {"role": "system", "content": "The text I deliver is a set of words in the form of a list, and please combine and guess the words to classify the class."},
                {"role": "user", "content": f"The text I want to convey is: {text}."},
                {"role": "assistant", "content": f"Please provide a classification: 1, 2, or 3 based on the content you just shared."}
            ]
            
        )
        return response.choices[0].message['content']
    
    def chatGPT_run(self, idx,df_report):
        print("chatGPT 프로세스...")

        n = len(df_report.loc[idx]['Crop_imgs'])
        crop_class_names = df_report.loc[idx]['Crop_classes']
        crop_texts = df_report.loc[idx]['ClovaOCR_text'].copy()

        # -1:초기화, 0:프레임, 1:공익, 2:정치 3:기타 
        categories = df_report.loc[idx]['Category']
        categories_basis = df_report.loc[idx]['Category_basis']
        for i in range(n):
            print("클래스명 : ", crop_class_names[i])
            if categories[i] == 0:
                print("내용 : *frame 입니다. (Detect 기반)")
                print(f"내용 : {crop_texts[i]} (카테고리: {0})")
                continue
            elif categories[i] == 1:
                print("내용 : *pulbic 입니다. (Detect 기반)")
                print(f"내용 : {crop_texts[i]} (카테고리: {1})")
            else:
                category_text = self.classify_text(crop_texts[i])  # OCR 텍스트를 GPT-3로 분류합니다.
                if 'Class 1' in category_text:
                    categories[i] = 1
                elif 'Class 2' in category_text:
                    categories[i] = 2
                else:
                    categories[i] = 3
                categories_basis[i] = category_text
                
        df_report.loc[idx]['Category'] = categories
        df_report.loc[idx]['Category_basis'] = categories_basis

        return df_report

    def run_all(self, imgs, json_file_path):
        df_report = self.make_frame()
        for idx,img in enumerate(imgs):
            # make_frame괄호속 파라미터 삭제
            time.sleep(1)
            df_report = self.yolo_run(img, df_report)
            time.sleep(1)
            # idx = 15 # 추정되는 인덱스, 필요에 따라 조절 가능.
            df_report = self.clova_ocr_run(idx, df_report)
            time.sleep(1)
            df_report = self.chatGPT_run(idx,df_report)

        # df_report.to_csv('./check_flask/views/reports/report_' + '_'.join(self.date_created) + '.csv')
        
        print("Process completed successfully.") 
        
        # return 추가
        return df_report
# # # 사용법
# analyzer = ImageProcess('best.pt', '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',
#                         'https://fsjr0lq9ke.apigw.ntruss.com/custom/v1/24396/82f04b3aebc287bf6b01f1571df49417fd2b38cb145fa7f9aadbb152eacbb606/general',
#                         'R1prcGNuRUthUG5hdGJPUW1Xd3pDVlVLUXdJZEx6UFM=')
# df_report = analyzer.run_all('IMAGE_PATH', 'JSON_PATH')




