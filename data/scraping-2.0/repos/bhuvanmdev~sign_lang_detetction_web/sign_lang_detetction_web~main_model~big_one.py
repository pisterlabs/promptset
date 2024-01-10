#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import copy
import itertools
from collections import Counter
import tensorflow as tf

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc

import openai
with open(r"C:\Users\bhuva\OneDrive\Desktop\hackman\hand-gesture-recognition-mediapipe\openaiapi.txt",'r') as f:
    openai.api_key = f.readlines()[0]

# lp = [r"C:\Users\bhuva\Downloads\my_trained_models\isl\keypoint(isl).csv",
#       r'model/keypoint_classifier/keypoint_classifier_label.csv',
#       r"C:\Users\bhuva\Downloads\my_trained_models\digits\keypoint_classifier_label(digits).csv"]
# mop = [r"C:\Users\bhuva\Downloads\my_trained_models\isl\keypoint_classifier(isl).tflite"
#     ,r'model/keypoint_classifier/keypoint_classifier.tflite'
#        ,r"C:\Users\bhuva\Downloads\my_trained_models\digits\keypoint_classifier(digits)fresh.tflite"]

class SIGN_MODEL:
    
    def __init__(self,model_path = r'model/keypoint_classifier/keypoint_classifier.tflite'
                ,label_path = r'model/keypoint_classifier/keypoint_classifier_label.csv'):
        self.loop,self.model_path,self.label_path = 0,model_path,label_path
        self.lp,self.mop = label_path,model_path
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path,
        num_threads = 1)
        self.real = True

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.use_brect = True

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )


        # Read labels ###########################################################
        with open(self.label_path,
                  encoding='utf-8-sig') as f:
            self.keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [
                row[0] for row in self.keypoint_classifier_labels
            ]

        # FPS Measurement ########################################################
        self.cvFpsCalc = CvFpsCalc(buffer_len=10)
        self.l = []
        self.text = ''
    def reset(self,mop,lp):
        self.loop,self.model_path,self.label_path = 0,mop,lp
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path,
        num_threads = 1)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()


        # Read labels ###########################################################
        with open(self.label_path,
                  encoding='utf-8-sig') as f:
            self.keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [
                row[0] for row in self.keypoint_classifier_labels
            ]
        self.real = False


    def KeyPointClassifier(
            self,
            landmark_list,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))

        self.interpreter.invoke()
        output_details_tensor_index = self.output_details[0]['index']
        #         print(self.output_details)
        result = self.interpreter.get_tensor(output_details_tensor_index)
        #         print(np.squeeze(result))
        result_index = np.argmax(np.squeeze(result))

        return result_index

    def calc_bounding_rect(self,image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)  #
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv.boundingRect(landmark_array)

        return [x, y, x + w, y + h]

    def draw_text(self,image):
        cv.putText(image, self.text, (300, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0))
        return image
#cv.FONT_HERSHEY_COMPLEX_SMALL,
    # claculates the landmarks (x,y)  with image size and returns the new landmarks in [[x,y],..] format
    def calc_landmark_list(self,image, landmarks):
        image_height_y, image_width_x, _ = image.shape

        landmark_point = []

        # Keypoint
        for landmark in landmarks.landmark:  # 21/42  loops for learning 1/2 hands
            landmark_x = min(int(landmark.x * image_width_x), image_width_x - 1)
            landmark_y = min(int(landmark.y * image_height_y), image_height_y - 1)
            # landmark_z = landmark.z

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    # the co-ordinates obtained from multiplying with screen size is normalized(0-1 range) and relative co-ordinates here once again
    def pre_process_landmark(self,landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        # Convert to relative coordinates(having a x=0 and y=0)
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Convert to a one-dimensional list
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))  # convert them from 0-1 range

        return temp_landmark_list

    def draw_bounding_rect(self, image, brect):
        if self.use_brect:
            # Outer rectangle
            cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                         (0, 0, 0), 1)

        return image

    def draw_info_text(self,image, brect, handedness, hand_sign_text ):
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                     (0, 0, 0), -1)
        info_text = handedness.classification[0].label[0:]
        if (self.keypoint_classifier_labels)[hand_sign_text] != "":
            info_text = info_text + ':' + (self.keypoint_classifier_labels)[hand_sign_text]
        cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        return image

    def draw_info(self,image, fps):
        cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (255, 255, 255), 2, cv.LINE_AA)
        return image



    def model_runner(self,image):
        fps = self.cvFpsCalc.get()

        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)


        #very important bhuvan
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True
    #         print(keypoint_classifier_labels)
        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness): # 1/2 loops
                # Bounding box calculation
                self.mp_drawing.draw_landmarks(
                    debug_image,  # image to draw
                    hand_landmarks,  # model output
                    self.mp_hands.HAND_CONNECTIONS,  # hand connections
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())

                brect = self.calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = self.calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = self.pre_process_landmark(
                    landmark_list)


                # Hand sign classification the mik(id=label in  keypoint.csv)
                hand_sign_id = self.KeyPointClassifier(pre_processed_landmark_list)
                char = self.keypoint_classifier_labels[hand_sign_id]
                self.l.append(char)

                if len(self.l) == 30:
                    char_final = Counter(self.l).most_common()


                    if char_final[0][0] == 'enter':
                        k = self.text
                        openai.ChatCompletion.create(model='gpt-3.5-turbo',
                                                            messages=[{"role":'system','content':f'if the english sentence in quotes sounds meaningless then correct it else return it without change"""{self.text}"""'}]
                                                            ,temperature=0.6)["choices"][0]["message"]["content"]
                        self.text = ''
                        self.l.clear()
                        return k,cv.resize(debug_image, (500, 400))
                        # print(fin)
                    else:
                        self.text += char_final[0][0]
                    self.l.clear()
                debug_image = self.draw_text(debug_image)
                debug_image = self.draw_bounding_rect(debug_image, brect)
                debug_image = self.draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    hand_sign_id
                )
        else:
            if self.loop==30:
                self.loop = 0
                self.l.clear()
            else:
                self.loop += 1
                # Drawing part

    #                 print(landmark_list) #bhuvan

        debug_image = self.draw_info(debug_image, fps)
        return '',cv.resize(debug_image, (500, 400))
    # Screen reflection #############################################################
    # cv.imshow('Hand Gesture Recognition', debug_image)
#         sleep(5)#bhuvan

# cap.release()
# cv.destroyAllWindows()


