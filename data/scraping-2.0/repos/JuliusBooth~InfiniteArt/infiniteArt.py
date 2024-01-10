import os
import openai
from secrets import *


from PIL import Image
import requests
from io import BytesIO
import cv2
import numpy as np
import time


def get_image_variation(source):
    response = openai.Image.create_variation(
      image=open(source, "rb"),
      n=1,
      size="1024x1024"
    )
    return response['data'][0]['url']

def save_image_from_url(url, save_location):
    response = requests.get(url)
    with open(save_location, 'wb') as handler:
        handler.write(response.content)

def create_variant(image_name, last_image, i):
    input_url = "images/{}/input.png".format(image_name)
    cv2.imwrite(input_url, last_image)
    variant_url = get_image_variation(input_url)
    new_file_name = "images/{0}/{0}{1}.png".format(image_name, i)
    save_image_from_url(variant_url, new_file_name)
    new_image = cv2.imread(new_file_name)
    return new_image


def create_altered_variant(image_name, last_image, average_image, i):
    cv2.imwrite("images/{}/average.png".format(image_name), average_image)

    # get black and white versions of last_image and average_image
    last_image_bw = cv2.cvtColor(last_image, cv2.COLOR_BGR2GRAY)
    average_image_bw = cv2.cvtColor(average_image, cv2.COLOR_BGR2GRAY)
    # get the absolute difference between the two
    difference = cv2.absdiff(last_image_bw, average_image_bw)
    difference = cv2.cvtColor(difference, cv2.COLOR_GRAY2BGR)
    # smooth difference
    difference = cv2.GaussianBlur(difference, (5,5), 0)

    average_image = cv2.addWeighted(average_image, 0.05, last_image, 0.95, 0)

    cv2.imwrite("images/{}/difference.png".format(image_name), difference)
    random_image = np.random.randint(0, 255, size=difference.shape, dtype=np.uint8)
    threshold = 5
    if i % 5 == 0:
        threshold = 80

    threshold = np.random.randint(0, threshold)
    adjusted_image = np.where(difference < threshold, random_image, last_image)

    adjusted_image_url = "images/{}/adjusted.png".format(image_name)
    cv2.imwrite(adjusted_image_url, adjusted_image)
    
    new_image = create_variant(image_name, adjusted_image, i)

    return new_image, average_image


def get_variants(image_name, n):
    starting_file_name = "images/{0}/{0}.png".format(image_name)
    starting_image = cv2.imread(starting_file_name)
    new_image = starting_image
    average_image = np.random.randint(0, 255, size=new_image.shape, dtype=np.uint8)

    for i in range(n):
        time.sleep(1)
        #new_image, average_image = create_altered_variant(image_name, new_image, average_image, i)
        new_image = create_variant(image_name, new_image, i)

def convert_avi_to_mp4(avi_file_path, output_name):
    os.popen("ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 'videos/{output}.mp4'".format(input = avi_file_path, output = output_name))
    time.sleep(5)
    os.remove(avi_file_path)
    return True

def resize_and_save_image(image_name, image_size=(1024,1024)):
    template_name = "templates/{}.jpeg".format(image_name)
    image = cv2.imread(template_name)
    resized=cv2.resize(image,image_size)
    new_directory = "images/{}".format(image_name)
    if not os.path.isdir(new_directory):
        os.mkdir(new_directory)
    new_file_name = new_directory + "/{}.png".format(image_name)
    cv2.imwrite(new_file_name, resized)

def get_random_photo():
    url = "https://source.unsplash.com/random/1024x1024"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img.save('random.png')
    #TODO: use random photo as template

def make_video(image_name):
    img_array = []
    size = (512,512)
    i = 0
    while True:
        try:
            filename = 'images/{0}/{0}{1}.png'.format(image_name,i)
            i+=1
            img = cv2.imread(filename)
            resized=cv2.resize(img,size)
            img_array.append(resized)
        except:
            print("ran out of images at" + filename)
            break

    first_image = cv2.imread("images/{0}/{0}.png".format(image_name))
    resized=cv2.resize(first_image,size)
    img_array.insert(0, resized)

    video_name = image_name + '.avi'
    video = cv2.VideoWriter(video_name, 0, 8, size)

    for image in img_array:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()
    convert_avi_to_mp4(video_name, image_name)

def make_video_from_template(image_name, n=100):
    resize_and_save_image(image_name)
    get_variants(image_name, n)
    make_video(image_name)

if __name__ == "__main__":
    make_video_from_template("virus", 50)
