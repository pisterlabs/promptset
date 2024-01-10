import openai
import time
from PIL import Image, ImageFont, ImageDraw
from gtts import gTTS

import random
from moviepy.editor import *
import ffmpeg
import numpy as np




# Vidio creating
openai.api_key = ""

def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=64,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text
folder = "C:\\Users\\FelixEdenborgh\\Documents\\PythonPrograms\\CreateTiktokVideosAndAutoUpload\\files"



def makeTxtFile():
    number = 1  # changed to start at 1
    name = f"Daily motivation quote {number}.txt"
    prompt = "Can you give me a motivational quote about love?"
    response = generate_response(prompt)
    print(response)

    file_path = os.path.join(folder, name)  # using os.path.join to handle path separator
    with open(file_path, "w") as file:
        file.write(response)
    print(f"File saved at {file_path}")
    time.sleep(2)





def ConvertToMp3():
    # path of folder containing input text files
    input_folder = folder
    print(input_folder)

    # path of folder to save output mp3 files
    output_folder = "C:\\Users\\FelixEdenborgh\\Documents\\PythonPrograms\\CreateTiktokVideosAndAutoUpload\\mp3"

    # get list of text files in input folder
    input_files = os.listdir(input_folder)
    print(input_files)

    try:
        # open first text file in folder
        file = open(os.path.join(input_folder, input_files[0]), 'r')
        try:
            # read text from file
            text = file.read()
            # convert text to speech
            tts = gTTS(text, lang='en', tld='com')
        finally:
            # close the file explicitly
            file.close()

        # save speech as mp3 file in output folder
        output_file = os.path.join(output_folder, input_files[0][:-4] + '.mp3')
        tts.save(output_file)

    except IOError as e:
        print("Error: {}".format(e))



def get_max_line_length(lines, font):
    max_length = 0
    for line in lines:
        length = font.getlength(line)
        if length > max_length:
            max_length = length
    return max_length

def get_random_image_from_folder():
    folder_path = "C:\\Users\\FelixEdenborgh\\Documents\\PythonPrograms\\CreateTiktokVideosAndAutoUpload\\Images"
    image_files = [file for file in os.listdir(folder_path) if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    if not image_files:
        return None  # No image files found in the folder

    random_image_file = random.choice(image_files)
    random_image_path = os.path.join(folder_path, random_image_file)

    return random_image_path

def get_average_brightness(image):
    img_array = np.array(image)
    avg_brightness = np.mean(img_array)
    return avg_brightness


def CreateAImageWithTheDailyQuoteWrittenOnIt():
    # Define input and output folders
    input_folder = folder
    output_folder = "C:\\Users\\FelixEdenborgh\\Documents\\PythonPrograms\\CreateTiktokVideosAndAutoUpload\\img"

    # Define font and font size
    font = ImageFont.truetype("arial.ttf", 36)

    # Define max width and height of text area
    max_width, max_height = 600, 600

    # Loop over input files in input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            # Read text from file
            with open(os.path.join(input_folder, filename), 'r') as file:
                text = file.read()

            # Get a random image to use as background
            random_image_path = get_random_image_from_folder()
            if random_image_path:
                image = Image.open(random_image_path)
            else:
                image = Image.new(mode='RGB', size=(800, 600), color=(255, 255, 255))

            draw = ImageDraw.Draw(image)

            # Wrap text based on max width and height
            words = text.split(' ')
            lines = []
            line = ''
            for word in words:
                if draw.textbbox((0, 0), line + word, font=font)[2] <= max_width and \
                        draw.multiline_textbbox((0, 0), line + word, font=font)[3] <= max_height:
                    line += word + ' '
                else:
                    lines.append(line)
                    line = word + ' '
            lines.append(line)

            # Get average brightness of the image
            avg_brightness = get_average_brightness(image)

            # Choose a text color based on image brightness
            if avg_brightness < 128:
                # Brighter text for dark images
                text_color_range = (200, 255)
            else:
                # Darker text for bright images
                text_color_range = (0, 100)

            # Write wrapped text to image
            x, y = 100, 100
            for line in lines:
                # Generate a random color for the text
                text_color = (
                random.randint(*text_color_range), random.randint(*text_color_range), random.randint(*text_color_range))
                draw.multiline_text((x, y), line, font=font, fill=text_color)
                y += draw.textbbox((x, y), line, font=font)[3]

            # Save image to output folder
            output_filename = os.path.splitext(filename)[0] + ".jpg"
            output_path = os.path.join(output_folder, output_filename)
            image.save(output_path)
            file.close()

CreateAImageWithTheDailyQuoteWrittenOnIt()


def CombineImageWithMp3(number):
    # Replace the placeholder with the path to your folder
    folder_path = "C:\\Users\\FelixEdenborgh\\Documents\\PythonPrograms\\CreateTiktokVideosAndAutoUpload\\mp3\\"

    # Get the list of file names in the folder
    file_names_mp3 = os.listdir(folder_path)

    # Get the first file name in the folder (or print a message if the folder is empty)
    if len(file_names_mp3) > 0:
        first_file_name_mp3 = file_names_mp3[0]
        print("The first file name in the folder is:", first_file_name_mp3)
    else:
        print("The folder is empty.")

    # Replace the placeholder with the path to your folder
    folder_path = "C:\\Users\\FelixEdenborgh\\Documents\\PythonPrograms\\CreateTiktokVideosAndAutoUpload\\img"

    # Get the list of file names in the folder
    file_names_image = os.listdir(folder_path)

    # Get the first file name in the folder (or print a message if the folder is empty)
    if len(file_names_image) > 0:
        first_file_name_image = file_names_image[0]
        print("The first file name in the folder is:", first_file_name_image)
    else:
        print("The folder is empty.")


    # Replace the placeholders with the paths to your image and mp3 files
    image_path = "C:\\Users\\FelixEdenborgh\\Documents\\PythonPrograms\\CreateTiktokVideosAndAutoUpload\\img\\" + first_file_name_image
    mp3_path = "C:\\Users\\FelixEdenborgh\\Documents\\PythonPrograms\\CreateTiktokVideosAndAutoUpload\\mp3\\" + first_file_name_mp3

    string_movie = str(number)
    # Replace the placeholders with the desired output file name and extension
    output_path = f"C:\\Users\\FelixEdenborgh\\Documents\\PythonPrograms\\CreateTiktokVideosAndAutoUpload\\Movie\\{string_movie}.mp4"

    # Load the image file using ffmpeg
    input_image = ffmpeg.input(image_path)

    # Load the audio file using ffmpeg
    input_audio = ffmpeg.input(mp3_path)

    # Create the video by overlaying the image over a black background and adding the audio
    video = ffmpeg.output(input_image, input_audio, output_path, vcodec='libx264', acodec='copy', strict='experimental')

    # Run the ffmpeg command to create the video
    ffmpeg.run(video)

