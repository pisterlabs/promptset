import cv2
import mediapipe as mp
import os
import pickle
import numpy as np
import time
import openai
import sys
from test_classifier import get_directory, open_model

def recognize_letter(model, target_letter):
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    labels_dict = {i: chr(65 + i) for i in range(26)}  # Mapping integers to alphabet letters

    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            sys.exit("Failed to grab frame")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        data_loc = []
        x_ = []
        y_ = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i, landmark in enumerate(hand_landmarks.landmark):
                    x, y = landmark.x, landmark.y
                    data_loc.extend([x, y])
                    x_.append(x)
                    y_.append(y)

            # Generate prediction from the model
            prediction = model.predict([np.asarray(data_loc)])
            predicted_character = labels_dict[int(prediction[0])]

            # Display the prediction on the frame
            cv2.putText(frame, predicted_character, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the target letter on the frame
        cv2.putText(frame, f"Show letter: {target_letter}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("frame", frame)

        elapsed_time = time.time() - start_time
        if elapsed_time > 10:  # 10-second time limit
            break

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Determine if the prediction was correct
    is_correct = predicted_character.lower() == target_letter.lower()
    return is_correct

def play_game(model):
    # Example game loop - here we would randomly select a letter and prompt the user
    target_letter = 'a'  # For testing, we're using 'a'. Implement random letter selection for the actual game.
    print(f"Please sign the letter: {target_letter}")

    correct = recognize_letter(model, target_letter)
    if correct:
        feedback = "Correct! Well done."
    else:
        feedback = "Oops! That was not correct. Try again."

    print(feedback)
    return feedback

def main():
    SCRIPT_DIR, DATA_DIR = get_directory()
    model = open_model(SCRIPT_DIR, DATA_DIR)
    play_game(model)

if __name__ == "__main__":
    main()