# import the necessary packages
import io
import openai
import nltk
import requests
import pytesseract
import imageio
import moviepy.editor as mp
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
from os import path

class Animator:
    def __init__(self, debug=False):
        self.debug = debug
        self.image = None
        key =  os.environ.get('OPENAI_API_KEY')
        if self.debug:
            print(key)
        if key == None:
            print("Error: OpenAI API key not found")
            exit(0)

        openai.api_key = key

    def get_text(self, image_name):
        
        # read the image
        self.image = cv2.imread(image_name, cv2.IMREAD_COLOR)

        print(">Done reading image")
        if self.debug:
            cv2.imshow("Image", self.image)
            cv2.waitKey(0)

        # convert the image to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        print(">Done converting to grayscale")
        if self.debug:
            cv2.imshow("Gray", gray)
            cv2.waitKey(0)

        # Apply adaptive thresholding to the grayscale image
        thresholded = cv2.adaptiveThreshold(gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 4)#17, 4
        print(">Done applying adaptive thresholding")
        if self.debug:
            cv2.imshow("Gaussian Adaptive Thresholding", thresholded)
            cv2.waitKey(0)

        # Use pytesseract to extract text from the preprocessed image
        text = pytesseract.image_to_string(thresholded, config='--psm 6')
        # Print the text
        if self.debug:
            print("\n"+text)

        return text

    def get_text_v2(self, image_name):
        
        # read the image
        self.image = cv2.imread(image_name, cv2.IMREAD_COLOR)

        print(">Done reading image")
        if self.debug:
            cv2.imshow("Image", self.image)
            cv2.waitKey(0)

        # convert the image to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        print(">Done converting to grayscale")
        if self.debug:
            cv2.imshow("Gray", gray)
            cv2.waitKey(0)

        # blur the image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        print(">Done blurring")
        if self.debug:
            cv2.imshow("Blurred", blurred)
            cv2.waitKey(0)

        # erode the image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded = cv2.erode(blurred, kernel, iterations=1)
        print(">Done erooding")
        if self.debug:
            cv2.imshow("Eroded", eroded)
            cv2.waitKey(0)

        #dilate the image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(eroded, kernel, iterations=1)
        print(">Done dilating")
        if self.debug:
            cv2.imshow("Dilated", dilated)
            cv2.waitKey(0)

        # opening
        kernelSize = (1,1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
        opening = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel)
        print(">Done opening")
        if self.debug:
            cv2.imshow("Opened", opening)
            cv2.waitKey(0)

        # Apply adaptive thresholding to the grayscale image
        thresholded = cv2.adaptiveThreshold(opening, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 4)#17, 4
        print(">Done applying adaptive thresholding")
        if self.debug:
            cv2.imshow("Gaussian Adaptive Thresholding", thresholded)
            cv2.waitKey(0)

        # Use pytesseract to extract text from the preprocessed image
        text = pytesseract.image_to_string(thresholded, config='--psm 6')
        # Print the text
        if self.debug:
            print("\n"+text)

        return text

    # # Define the callback function for mouse events
    # def crop_image(self, event, x, y, flags, param):
    #     global x_start, y_start, x_end, y_end, cropping, crop

    #     # If the left mouse button is pressed, start cropping
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         x_start, y_start = x, y
    #         x_end, y_end = x, y
    #         cropping = True

    #     # If the mouse is moving while cropping, update the end coordinates
    #     elif event == cv2.EVENT_MOUSEMOVE:
    #         if cropping:
    #             x_end, y_end = x, y

    #     # If the left mouse button is released, stop cropping and crop the image
    #     elif event == cv2.EVENT_LBUTTONUP:
    #         x_end, y_end = x, y
    #         cropping = False

    #         # Crop the image to the selected ROI
    #         crop = self.image[min(y_start, y_end):max(y_start, y_end), min(x_start, x_end):max(x_start, x_end)]

    def generate_image(self, prompt="", no_of_images=1, size="256x256",image_name="image.jpg"):
        # no_of_images = -int- [1-10]
        # size = -string- [256x256, 512x512, 1024x1024] 

        # Send the text to the server
        # create the image sending http post request and get the response
        try:
            response = openai.Image.create(
                prompt=prompt,
                n=no_of_images,
                size=size
            )

            # get the image url
            image_url = response['data'][0]['url']

        except openai.error.OpenAIError as e:
            print("Error: ", e)
            return
        
        if image_url == None:
            print("Error: Image url is none")
            return
        else:
            print("Image url: ", image_url)

        # download the image
        # Download the image from the URL
        image_data = requests.get(image_url).content

        # Open the image using PIL
        image = Image.open(io.BytesIO(image_data))

        # Save the image as a JPEG file
        image.save(image_name)

    def tokenize_sentences(self, paragrapgh):
        # Tokenize the paragraph into sentences and remove the newline character
        sentences = nltk.tokenize.sent_tokenize(paragrapgh)
        sentences = [sentence.replace('\n', '') for sentence in sentences]

        return sentences

    def generate_video(self, sentences, output_video_path, duration, fps, subtitles=False):

        # Initialize a list to store the images
        frames = []

        # Generate frames with text
        for i in range(0, len(sentences)):
            self.generate_image(prompt=sentences[i]+"realistic", no_of_images=1, size="256x256", image_name="output_"+str(i)+".jpg")

            #append the image to the frames list
            frames.append("output_"+str(i)+".jpg")

        # Save frames as individual images
        for i, frame in enumerate(frames):
            image = imageio.imread(frame)
            imageio.imwrite(f"frame_{i}.png", image)

        # Create the video using the frames
        with imageio.get_writer(output_video_path, fps=fps) as writer:
            for i in range(len(frames)):
                image = imageio.imread(f"frame_{i}.png")
                writer.append_data(image)

        # Convert the video to the desired format (optional)
        output_video = mp.VideoFileClip(output_video_path)
        #output_video.write_videofile("final_video.mp4")

        # Clean up the temporary image files
        for i in range(len(frames)):
            os.remove(f"frame_{i}.png")

        print("Video created successfully.")
