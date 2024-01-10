"""
This program generates photos from random audio conversations and displays 
them on the screen. When run on command, it is an interesting exercise in the power of OpenAI.
When run in continuous mode, it is a creepy photo generator because it shows how good openAI is 
at understanding what you are saying.

There is a step that is now commented out that summarizes the transcript. The summary is 
errily accurate.

Basic flow:
    * record audio from the default microphone and then transcribe it using OpenAI
    * summarize the transcript and generate 4 pictures based on the summary
    * combine the four images into a single image
    * open the picture in a browser
    * optionally, delay for 60 seconds and repeat the process
    * images are stored in the history directory

The program can be run in two modes:
    
1/  python3 pyspeech.py
    This will display a menu and prompt you for a command. 
2/  python3 pyspeech.py -h
    For testing. Use command line arguments  

control-c to stop the program. When run in auto mode it will loop 10 times

For debug output, use the -d 2 argument. This will show the prompts and responses to/from OpenAI.

To run this you need to get an OpenAI API key and put it in a file called "creepy photo secret key".
OpenAI currently costs a few pennies to use. I've run this for an hour at a cost of $1.00. It was
well worth it.

ALSO NOTE: If you are not getting any audio, then you may not have given the program
permission to access your microphone. On OSX it took me some searching to figure this out.
https://superuser.com/questions/1441270/apps-dont-show-up-in-camera-and-microphone-privacy-settings-in-macbook
Until the Terminal app showed up in Settings / Privacy & Security / Microphone this program
just wont work. 
On the RPi I had to add my user to the "audio" group. I did this
with      usermod -a -G audio <username>

Based on the WhisperFrame project idea on Hackaday.
https://hackaday.com/2023/09/22/whisperframe-depicts-the-art-of-conversation/

Specific to Raspberry Pi:
    0. clone repo
       git clone https://github.com/jschrempp/speech2picture.git speech2picture

    1. set up a virtual environment and activate it (to deactive use "deactivate")
        cd speech2picture
        python3 -m venv .venv
        source .venv/bin/activate

        set your openai key
            https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety

            nano ~/.bashrc and comment out these lines
                # If not running interactively, don't do anything
                case $- in
                    *i*) ;;
                    *) return;;
                esac

            Then add this line
                export OPENAI_API_KEY='yourkey'

            Check your work
                source ~/.bashrc
                echo $OPENAI_API_KEY

    2. install the following packages

        2a. for RPi version 3 install these
            sudo apt-get install portaudio19-dev
            On the 2023-10-10 64 bit Raspbian OS you don't need to install these
            #sudo apt-get install libasound2-dev
            #sudo apt-get install libatlas-base-dev
            #sudo apt-get install libopenblas-dev

            cp s2p.desktop ~/Desktop

            cd ..
            mkdir .config/lxsession
            mkdir .config/lxsession/LXDE-pi
            mkdir .config/lxsession/LXDE-pi/autostart
            cp Desktop/s2p.desktop .config/lxsession/LXDE-pi/autostart/s2p.desktop

            OR

            sudo cp Desktop/s2p.desktop /usr/share/xsessions/s2p.desktop

        2b. on MacOS install these
            brew install portaudio
            brew update-reset   # if your brew isn't working try this
            xcode-select --install  # needed for pyaudio to install
            pip3 install sounddevice
            pip3 install soundfile
            pip3 install numpy

            Use finder and navigate to /Applications/Python 3.12
                  Then doublelick on "Install Certificates.command"

                    
    3. install the following python packages    
        pip install openai 
        pip install pillow
        pip install pyaudio
        pip install RPi.GPIO

    Note that when run you will see 10 or so lines of errors about sockets and JACKD and whatnot.
    Don't worry, it is still working. If you know how to fix this, please let me know.

    Also note that errors from the audio subsystem are ignored in recordAudioFromMicrophone(). If 
    you are having some real audio issue, you might change the error handler to print the errors.

    If you want to make this run on boot, then see the comments in s2p.desktop
    
Author: Jim Schrempp 2023 

Version History of significant changes:

v 0.8 added "without any text or writing in the image" to the image prompt
v 0.7 more code cleanup, improved image resizing for display size
      added QR code
v 0.6 added -g for gokiosk mode
v 0.5 Initial version
v 0.6 2023-11-12 inverted Go Button logic so it is active low (pulled to ground)
v 0.7 updated to python 3.12 and openAI 1.0.0 (wow that was a pain)
      BE SURE to read updated install instructions above
"""

# import common libraries
import platform
import argparse
import logging
from logging.handlers import TimedRotatingFileHandler
import urllib.request
import time
import shutil
import re
import os
import random
import tkinter as tk
from enum import IntEnum
from PIL import Image, ImageDraw, ImageFont, ImageTk

import openai

g_isMacOS = False
if (platform.system() == "Darwin"):
    g_isMacOS = True
else:
    print ("Not MacOS")

# import platform specific libraries
if g_isMacOS:
    import sounddevice
    import soundfile

else:
    # --------- import for Raspberry Pi -----------------------------------------
    import pyaudio
    import wave
    from ctypes import *
    import RPi.GPIO as GPIO
    import threading
    from queue import Queue





# Global reference to the windows
# need to be outside the global class so that tkinter can access them
g_windowForImage = None
g_windowForInstructions = None

# Global constants
LOOPS_MAX = 10 # Set the number of times to loop when in auto mode

# Prompt for abstraction
PROMPT_FOR_ABSTRACTION = "What is the most interesting concept in the following text \
    expressing the answer as a noun phrase, but not in a full sentence "

# image prompt modifiers
# 'generate a picture [MODIFIER] for the following concept: ...'
IMAGE_MODIFIERS = [
    "as a painting by Picasso",
    "as a watercolor by Picasso",
    "as a sketch by Picasso",
    "as a vivid color painting by Monet",
    "as a painting by Van Gogh",
    "as a painting by Dali",
    "in the style of Escher",
    "in the style of Rembrandt",
    "as a photograph by Ansel Adams",
    "as a painting by Edward Hopper",
    "as a painting by Norman Rockwell",
]

# Define  constants for blinking the LED (onTime, offTime)
BLINK_FAST = (0.1, 0.1)
BLINK_SLOW = (0.5, 0.5)
BLINK_FOR_AUDIO_CAPTURE = (0.05, 0.05)
BLINK1 = (0.5, 0.2)
BLINK2 = (0.4, 0.2)
BLINK3 = (0.3, 0.2)
BLINK4 = (0.2, 0.2)
BLINK_STOP = (-1, -1)
BLINK_DIE = (-2, -2)

if not g_isMacOS:
    # Define the GPIO pins for RPi
    LED_RED = 8
    BUTTON_GO = 10
    BUTTON_PULL_UP_DOWN = GPIO.PUD_UP
    BUTTON_PRESSED = GPIO.LOW  

# used by command line args to jump into the middle of the process
class processStep(IntEnum):
        NoneSpecified = 0
        Audio = 1
        Transcribe = 2
        Summarize = 3
        Keywords = 4
        Image = 5

 # global variables
class g_vars:
   
    # Set the duration of each recording in seconds
    duration = 120

    # if true don't use the command menu if we're using a button
    isUsingHardwareButtons = False  

    # When true don't extract keywords from the transcript, just use it for the image prompt
    isAudioKeywords = False

    # when running continuous, this will limit the actual number of iterations
    numLoops = 1

    # when running continuous, this will delay between iterations
    loopDelay = 0

    # command line arguments can set this to jump into the middle of the process
    firstProcessStep = processStep.Audio

    # if command line args specify to use a file, then set this to it
    inputFileName = None

    # if true, then save files that are generated in the process - mostly a debug feature
    isSaveFiles = False
    
g = g_vars()

# XXX client = OpenAI()  # must have set up your key in the shell as noted in comments above
client = openai

# set up logging
logger = logging.getLogger(__name__) # parameter: -d 1
loggerTrace = logging.getLogger("Prompts") # parameter: -d 2
logging.basicConfig(level=logging.WARNING, format=' %(asctime)s - %(levelname)s - %(message)s')

logToFile = logging.getLogger("s2plog")
logToFile.setLevel(logging.INFO)
handler = TimedRotatingFileHandler('s2plog.log', when="midnight", interval=7, backupCount=10)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logToFile.addHandler(handler)


# create root window for display and hide it
root = tk.Tk()
root.withdraw()  # Hide the root window


if not g_isMacOS:
    # --------- Raspberry Pi specific code -----------------------------------------
    logger.info("Setting up GPIO pins")

    # Set the pin numbering mode to BCM
    GPIO.setmode(GPIO.BOARD)

    # Set up pin g.LEDRed as an output
    GPIO.setup(LED_RED, GPIO.OUT, initial=GPIO.LOW)
    
    # Set up pin 10 as an input for the start button
    GPIO.setup(BUTTON_GO, GPIO.IN, pull_up_down=BUTTON_PULL_UP_DOWN)

    # Define a function to blink the LED
    # This function is run on a thread
    # Communicate by putting a tuple of (onTime, offTime) in the qBlinkControl queue
    #
    def blink_led(q):
        # print("Starting LED thread") # why do I need to have this for the thread to work?
        logger.info("logging, Starting LED thread")

        # initialize the LED
        isBlinking = False
        GPIO.output(LED_RED, GPIO.LOW)

        while True:
            # Get the blink time from the queue
            try:
                blink_time = q.get_nowait()
            except:
                blink_time = None

            if blink_time is None:
                # no change
                pass
            elif blink_time[0] == -2:
                # die
                logger.info("LED thread dying")
                break
            elif blink_time[0] == -1:
                # stop blinking
                GPIO.output(LED_RED, GPIO.LOW)
                isBlinking = False
            else:
                onTime = blink_time[0]
                offTime = blink_time[1]
                isBlinking = True

            if isBlinking:
                # Turn the LED on
                GPIO.output(LED_RED, GPIO.HIGH)
                # Wait for blink_time seconds
                time.sleep(onTime)
                # Turn the LED off
                GPIO.output(LED_RED, GPIO.LOW)
                # Wait for blink_time seconds
                time.sleep(offTime)

    # Create a new thread to blink the LED
    logger.info("Creating LED thread")
    qBlinkControl = Queue()
    led_thread1 = threading.Thread(target=blink_led, args=(qBlinkControl,),daemon=True)
    led_thread1.start()


    # --------- end of Raspberry Pi specific code ----------------------------


def changeBlinkRate(blinkRate):
    '''change the LED blink rate. This routine isolates the RPi specific code'''
    if not g_isMacOS:
        # running on RPi
        qBlinkControl.put(blinkRate)
    else:
        # not running on RPI so do nothing
        pass


def recordAudioFromMicrophone():
    '''record duration seconds of audio from the default microphone to a file and return the sound file name'''

    soundFileName = 'recording.wav'
    
    # delete file recording.wav if it exists
    try:
        os.remove(soundFileName)
    except:
        pass # do nothing   
    
    if g_isMacOS:
        # print the devices
        # print(sd.query_devices())  # in case you have trouble with the devices

        # Set the sample rate and number of channels for the recording
        sample_rate = int(sounddevice.query_devices(1)['default_samplerate'])
        channels = 1

        logger.debug('sample_rate: %d; channels: %d', sample_rate, channels)

        logger.info("Recording %d seconds...", g.duration)
        os.system('say "Recording."')
        # Record audio from the default microphone
        recording = sounddevice.rec(
            int(g.duration * sample_rate), 
            samplerate=sample_rate, 
            channels=channels
            )

        # Wait for the recording to finish
        sounddevice.wait()

        # Save the recording to a WAV file
        soundfile.write(soundFileName, recording, sample_rate)

        os.system('say "Thank you. I am now analyzing."')

    else:

        # RPi
        """
        recording = sounddevice.Stream(channels=1, samplerate=44100)
        recording.start()
        time.sleep(15)
        recording.stop()
        soundfile.write('test1.wav',recording, 44100)
        """

        # all this crap because the ALSA library can't police itself
        ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
        def py_error_handler(filename, line, function, err, fmt):
            pass #nothing to see here
        c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
        asound = cdll.LoadLibrary('libasound.so')
        # Set error handler
        asound.snd_lib_error_set_handler(c_error_handler)
        # Initialize PyAudio
        pa = pyaudio.PyAudio()
        # Reset to default error handler
        asound.snd_lib_error_set_handler(None)
        # now on with the show, sheesh

        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            input=True,
            frames_per_buffer=1024
            ) #,input_device_index=2)

        wf = wave.open(soundFileName,"wb")
        wf.setnchannels(1)
        wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)

        # Write the audio data to the file
        for i in range(0, int(44100/1024*10)):

            # Get the audio data from the microphone
            data = stream.read(1024)

            # Write the audio data to the file
            wf.writeframes(data)

        # Close the microphone and the wave file
        stream.close()
        wf.close()

    return soundFileName


def getTranscript(wavFileName):
    '''transcribe the audio file and return the transcript'''

    # transcribe the recording
    logger.info("Transcribing...")
    audio_file= open(wavFileName, "rb")
    responseTranscript = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file)

    # print the transcript object
    loggerTrace.debug("Transcript object: " + str(responseTranscript))

    transcript = responseTranscript.text 

    loggerTrace.debug("Transcript text: " + transcript)

    return transcript


def getSummary(textInput):
    '''summarize the transcript and return the summary'''
    
    # summarize the transcript 
    logger.info("Summarizing...")

    responseSummary = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "user", "content" : 
                            f"Please summarize the following text:\n{textInput}" }
                        ])
    loggerTrace.debug("responseSummary: " + str(responseSummary))

    summary = responseSummary.choices[0].message.content.strip()
    
    logger.debug("Summary: " + summary)

    return summary


def getAbstractForImageGen(inputText):
    '''get keywords for the image generator and return the keywords'''

    # extract the keywords from the summary

    logger.info("Extracting...")
    logger.debug("Prompt for abstraction: " + PROMPT_FOR_ABSTRACTION)    

    prompt = PROMPT_FOR_ABSTRACTION + "'''" + inputText + "'''"
    loggerTrace.debug ("prompt for extract: " + prompt)

    responseForImage = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "user", "content": prompt}
                        ])

    loggerTrace.debug("responseForImageGen: " + str(responseForImage))

    # extract the abstract from the response
    abstract = responseForImage.choices[0].message.content.strip()
    
    # Clean up the response from OpenAI
    # delete text before the first double quote
    abstract = abstract[abstract.find("\"")+1:]
    # delete text before the first colon
    abstract = abstract[abstract.find(":")+1:]
    # eliminate phrases that are not useful for image generation
    badPhrases = ["the concept of", "in the supplied text is", "the most interesting concept"
                    "in the text is"]
    for phrase in badPhrases:
        # compilation step to escape the word for all cases
        compiled = re.compile(re.escape(phrase), re.IGNORECASE)
        res = compiled.sub(" ", abstract)
        abstract = str(res) 

    logger.info("Abstract: " + abstract)

    return abstract


def getImageURL(phrase):
    '''get images and return the urls'''

    # pick random modifiers
    random.shuffle(IMAGE_MODIFIERS)
  
    # create the prompt for the image generator
    prompt = f"Generate a picture {IMAGE_MODIFIERS[0]} without any text or writing in the image for the following concept: {phrase}"

    logger.info("Generating image...")
    logger.info("image prompt: " + prompt)

    # use openai to generate a picture based on the summary
    try:
        responseImage = client.images.generate(
            prompt= prompt,
            n=4,
            size="512x512")
    except Exception as e:
        print("\n\n\n")
        print(e)
        print("\n\n\n")
        raise (e)
        
    loggerTrace.debug("responseImage: " + str(responseImage))

    image_url = [responseImage.data[0].url] * 4
    image_url[1] = responseImage.data[1].url
    image_url[2] = responseImage.data[2].url
    image_url[3] = responseImage.data[3].url

    return image_url, IMAGE_MODIFIERS[0]


def postProcessImages(imageURLs, imageModifiers, keywords, timestr):
    '''reformat the images for display and return the new file name'''

    # save the images from a urls into imgObjects[]
    imgObjects = []
    for numURL in range(len(imageURLs)):

        fileName = "history/" + "image" + str(numURL) + ".png"
        urllib.request.urlretrieve(imageURLs[numURL], fileName)

        img = Image.open(fileName)

        imgObjects.append(img)

    # combine the images into one image
    #widths, heights = zip(*(i.size for i in imgObjects))
    total_width = 512*2
    max_height = 512*2 + 50
    new_im = Image.new('RGB', (total_width, max_height))
    locations = [(0,0), (512,0), (0,512), (512,512)]
    count = -1
    for loc in locations:
        count += 1
        new_im.paste(imgObjects[count], loc)

    # add text at the bottom
    imageCaption = f'{keywords} {imageModifiers}'
    draw = ImageDraw.Draw(new_im)
    draw.rectangle(((0, new_im.height - 50), (new_im.width, new_im.height)), fill="black")
    font = ImageFont.truetype("arial.ttf", 18)
    # decide if text will exceed the width of the image
    #textWidth, textHeight = font.getsize(text)
    draw.text((10, new_im.height - 30), imageCaption, (255,255,255), font=font)

    # save the combined image
    newFileName = "history/" + timestr + "-image" + ".png"
    new_im.save(newFileName)

    return newFileName


def generateErrorImage(e, timestr):
    '''generate an image with the error message and return the new file name'''

    # make an image to display the error
    total_width = 512*2
    max_height = 512*2 + 50
    new_im = Image.new('RGB', (total_width, max_height))
    draw = ImageDraw.Draw(new_im)
    draw.rectangle(((0, 0), (new_im.width, new_im.height)), fill="black")
    
    # add error text
    imageCaption = str(e)
    
    font = ImageFont.truetype("arial.ttf", 24)
    # decide if text will exceed the width of the image
    #textWidth, textHeight = font.getsize(text)

    import textwrap
    lines = textwrap.wrap(imageCaption, width=60)  #width is characters
    y_text = new_im.height/2
    for line in lines:
        #width, height = font.getsize(line)
        #draw.text(((new_im.width - width) / 2, y_text), line, font=font) 
        #y_text += height
        height = 25
        draw.text((100, y_text), line, font=font) 
        y_text += height

    #draw.text((10, new_im.height/2), imageCaption, (255,255,255), font=font)

    # save the new image
    newFileName = "errors/" + timestr + "-image" + ".png"
    new_im.save(newFileName)

    return newFileName

def create_instructions_window():
    '''create and display a window with static instructions'''

    global g_windowForInstructions

    # Instructions text
    INSTRUCTIONS_TEXT = ('\r\nWelcome to\rSpeech 2 Picture\n\rWhen you are ready, press and release the'
                    + ' button. You will have 10 seconds to speak your instructions. Then wait.'
                    + ' An image will appear shortly.'
                    + '\r\nUntil then, enjoy some previous images!')

    g_windowForInstructions = tk.Toplevel(root, bg='#52837D')
    g_windowForInstructions.title("Instructions")
    labelTextLong = tk.Label(g_windowForInstructions, text=INSTRUCTIONS_TEXT, 
                     font=("Helvetica", 28),
                     justify=tk.CENTER,
                     width=80,
                     wraplength=400,
                     bg='#52837D',
                     fg='#FFFFFF',
                     )
    g_windowForInstructions.minsize(200, 500)
    g_windowForInstructions.maxsize(500, 1000)
    g_windowForInstructions.geometry("+50+0") 

    frame = tk.Frame(g_windowForInstructions, bg='#52837D')

    labelQRText = tk.Label(frame, text="Scan this QR code for instructions on how to "
                           + "make your own speech to photo generator.", 
                     font=("Helvetica", 18),
                     justify=tk.LEFT,
                     wraplength=280,
                     bg='#52837D',
                     fg='#FFFFFF',
                     )

    # add the image to the window
    img = Image.open("S2PQR.png")
    img = img.resize((150,150), Image.NEAREST)
    photoImage = ImageTk.PhotoImage(img)
    label2 = tk.Label(frame, image=photoImage,
                     bg='#52837D')
    label2.image = photoImage  # Keep a reference to the image to prevent it from being garbage collected

    labelTextLong.pack(side=tk.TOP)
    frame.pack(fill=tk.X, pady=50)
    label2.pack(side=tk.LEFT,padx=20)
    labelQRText.pack(side=tk.LEFT,padx=10)

def create_image_window():
    '''create a window to display the images; return a label to display the images'''

    global g_windowForImage

    g_windowForImage = tk.Toplevel(root)
    g_windowForImage.title("Images")
    screen_width = g_windowForImage.winfo_screenwidth()
    screen_height = g_windowForImage.winfo_screenheight()
    g_windowForImage.geometry("+%d+%d" % (screen_width-1000, screen_height*.02))
    label = tk.Label(g_windowForImage)
    label.configure(bg='#000000')
    g_windowForImage.withdraw()  # Hide the window until needed

    return label

def display_image(image_path, label=None):
    '''display an image in the window using the label object'''

    global g_windowForImage

    if label is None:
        print("Error: label is None")  
        return

    # Open an image file
    try:
        img = Image.open(image_path)
    except Exception as e:
        print("Error opening image file")
        print(e)
        return

    #resize the image to fit the window
    screen_width = g_windowForImage.winfo_screenwidth()
    screen_height = g_windowForImage.winfo_screenheight()
    resizeFactor = 0.9 
    new_width = int(screen_height * resizeFactor * img.width / img.height)
    new_height = int(screen_height * resizeFactor)
    if img.width < 520:
        img = img.resize((new_width,new_height), Image.NEAREST)
    #images are typically 1024 x 1074   (1.05) (.95)
    elif img.height > screen_height-100:
        img = img.resize((new_width,new_height), Image.NEAREST)

    # Convert the image to a PhotoImage
    photoImage = ImageTk.PhotoImage(img)
    label.configure(image=photoImage)
    label.image = photoImage  # Keep a reference to the image to prevent it from being garbage collected
    label.pack() # Show the label
    g_windowForImage.deiconify() # Show the window now that it has an image

    return label

def display_random_history_image(labelForImageDisplay):
    '''display a random image from the history folder in the window using the label object'''

    global g_windowForImage

    # list all files in the history folder
    historyFolder = "./history"
    historyFiles = os.listdir(historyFolder)
    #remove any non-png files from historyFiles
    imagesToDisplay = []
    for file in historyFiles:
        if file.endswith(".png"):
            #add to the list
            imagesToDisplay.append(file)
    random.shuffle(imagesToDisplay) # randomize the list
    display_image(historyFolder + "/" + imagesToDisplay[0], labelForImageDisplay)
    
    # let the tkinter window events happen
    g_windowForImage.update_idletasks()
    g_windowForImage.update()


def close_image_window():
    '''close the image window'''

    global g_windowForImage

    if g_windowForImage is not None:
        g_windowForImage.destroy()
        g_windowForImage = None




def parseCommandLineArgs():
    '''parse the command line arguments and set the global variables'''

    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--savefiles", help="save the files", action="store_true") # optional argument
    parser.add_argument("-d", "--debug", help="0:info, 1:prompts, 2:responses", type=int) # optional argument
    parser.add_argument("-w", "--wav", help="use audio from file", type=str, default=0) # optional argument
    parser.add_argument("-t", "--transcript", help="use transcript from file", type=str, default=0) # optional argument
    parser.add_argument("-T", "--summary", help="use summary from file", type=str, default=0) # optional argument
    parser.add_argument("-k", "--keywords", help="use keywords from file", type=str, default=0) # optional argument
    parser.add_argument("-i", "--image", help="use image from file", type=str, default=0) # optional argument
    parser.add_argument("-o", "--onlykeywords", help="use audio directly without extracting keywords", action="store_true") # optional argument
    parser.add_argument("-g", "--gokiosk", help="jump into Kiosk mode", action="store_true") # optional argument
    args = parser.parse_args()

    # set the debug level
    logger.setLevel(logging.INFO)

    if args.debug == 1:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug level set to show prompts")
    elif args.debug == 2:
        logger.setLevel(logging.DEBUG)
        loggerTrace.setLevel(logging.DEBUG)
        logger.debug("Debug level set to show prompts and response JSON")


    # if true, don't ask user for input, rely on hardware buttons
    g.isUsingHardwareButtons = False

    if args.gokiosk:
        # jump into Kiosk mode
        print("\r\nKiosk mode enabled\r\n")
        g.isUsingHardwareButtons = True
        g.isAudioKeywords = True
        g.numLoops = 1
        g.loopDelay = 0
        g.firstProcessStep = processStep.NoneSpecified
    else:
        # if we're given a file via the command line then start at that step
        # check in reverse order so that processStartStep will be the latest step for any set of arguments
        g.firstProcessStep = processStep.NoneSpecified
        if args.image != 0: 
            g.firstProcessStep = processStep.Image
            g.inputFileName = args.image
        elif args.keywords != 0: 
            g.firstProcessStep = processStep.Keywords
            g.inputFileName = args.keywords
        elif args.summary != 0: 
            g.firstProcessStep = processStep.Summarize
            g.inputFileName = args.summary
        elif args.transcript != 0: 
            g.firstProcessStep  = processStep.Transcribe
            g.inputFileName = args.transcript
        elif args.wav != 0:
            g.firstProcessStep = processStep.Audio
            g.inputFileName = args.wav

        # if set, then record only 10 seconds of audio and use that for the keywords
        g.isAudioKeywords = False
        if args.onlykeywords:
            g.isAudioKeywords = True
            g.duration = 10


def main():
    # ----------------------
    # main program starts here
    #
    #
    # ----------------------

    #if not g_isMacOS:
        #Show instructions
    create_instructions_window()

    # create the window to display the images
    labelForImageDisplay = create_image_window()


    # create a directory if one does not exist
    if not os.path.exists("history"):
        os.makedirs("history")
    if not os.path.exists("errors"):
        os.makedirs("errors")


    parseCommandLineArgs()

    # if true, we had an error and want to just go back to the top of the loop
    g.abortThisIteration = False

    # ----------------------
    # Main Loop 
    #

    done = False  # set to true to exit the loop
    g.loopDelay = 60 # delay between loops in seconds

    randomDisplayMode = True 

    lastButtonPressedTime = 0
    lastImageDisplayedTime = 0

    while not done:

        if g.firstProcessStep > processStep.NoneSpecified:

            # we have file parameters, so only loop once
            g.numLoops = 1
            g.loopDelay = 1   # no delay if we're not looping XXX

        else:
            # no command line input parameters so get a command from the user

            inputCommand = None
            if not g.isUsingHardwareButtons: 
                # print menu
                print("\r\n\n\n")
                print("Commands:")
                print("   o: Once, record and display; default")
                print("   a: Auto mode, record, display, and loop")
                if not g_isMacOS:
                    # running on RPi
                    print("   h: Hardware control")
                print("   q: Quit")

                # BLOCKING CALL
                # wait for the user to press a key
                inputCommand = input("Type a command ...")

                if inputCommand == 'h':
                    # not in the menu except on RPi
                    # don't ask the user for input again, rely on hardware buttons
                    g.isUsingHardwareButtons = True
                    print("\r\nHardware control enabled")

                elif inputCommand == 'q': # quit
                    done = True
                    g.numLoops = 0
                    g.loopDelay = 0

                elif inputCommand == 'a': # auto mode
                    g.numLoops = LOOPS_MAX
                    print("Will loop: " + str(g.numLoops) + " times")
                    
                else: # default is once
                    g.numLoops = 1
                    g.loopDelay = 0
                    g.firstProcessStep = processStep.NoneSpecified

            # we can't use else here because the command menu input might set this value
            if g.isUsingHardwareButtons:
                # we're not going to prompt the user for input, rely on hardware buttons
                isButtonPressed = False

                while not isButtonPressed:
                    # running on RPi
                    # read gpio pin, if pressed, then do a cycle of keyword input
                    if GPIO.input(BUTTON_GO) == BUTTON_PRESSED:
                        g.isAudioKeywords = True
                        g.numLoops = 1
                        isButtonPressed = True
                        lastButtonPressedTime = time.time()
                        # print("stop random display " + str(lastButtonPressedTime))
                        randomDisplayMode = False

                    else:
                        # if the last button press was more than 90 seconds ago, then display history
                        if (time.time() - lastButtonPressedTime > 90) and (not randomDisplayMode):
                            # print ("start random display " + str(time.time()) + " " + str(lastButtonPressedTime))
                            lastImageDisplayedTime = time.time()
                            randomDisplayMode = True # stay in this mode until the button is pressed again
                            lastImageDisplayedTime = 0 # should display a picture immediately
                            
                        if randomDisplayMode:
                            if time.time() - lastImageDisplayedTime > 15:
                                display_random_history_image(labelForImageDisplay)
                                lastImageDisplayedTime = time.time()


        if g.isAudioKeywords: 
            # we are not going to extract keywords from the transcript
            g.duration = 10

        # we have a command. Either a command line file argument, a menu command, or a button press

        # loop will normally process audio and display the images
        # but if we're given a file then start at that step (processStep)
        # and numLoops should be 1
        for i in range(0, g.numLoops, 1):

            g.abortThisIteration = False

            # format a time string to use as a file name
            timestr = time.strftime("%Y%m%d-%H%M%S")

            soundFileName = ""
            transcript = ""
            summary = ""
            keywords = ""
            imageURLs = ""

            # Audio - get a recording.wav file
            if g.firstProcessStep <= processStep.Audio:

                changeBlinkRate(BLINK_FOR_AUDIO_CAPTURE)

                if g.firstProcessStep < processStep.Audio:
                    # record audio from the default microphone
                    soundFileName = recordAudioFromMicrophone()

                    if g.isSaveFiles:
                        #copy the file to a new name with the time stamp
                        shutil.copy(soundFileName, "history/" + timestr + "-recording" + ".wav")
                        soundFileName = "history/" + timestr + "-recording" + ".wav"
            
                else:
                    # use the file specified by the wav argument
                    soundFileName = g.inputFileName
                    logger.info("Using audio file: " + g.inputFileName)

                changeBlinkRate(BLINK_STOP)
        
            # Transcribe - set transcript
            if g.firstProcessStep <= processStep.Transcribe:
            
                changeBlinkRate(BLINK1)

                if g.firstProcessStep < processStep.Transcribe:
                    # transcribe the recording
                    transcript = getTranscript(soundFileName)
                    logToFile.info("Transcript: " + transcript)

                    if g.isSaveFiles:
                        f = open("history/" + timestr + "-rawtranscript" + ".txt", "w")
                        f.write(transcript)
                        f.close()
                else:
                    # use the text file specified 
                    transcriptFile = open(g.inputFileName, "r")
                    # read the transcript file
                    transcript = transcriptFile.read()
                    logger.info("Using transcript file: " + g.inputFileName)

                changeBlinkRate(BLINK_STOP)

            # Summary - set summary
            if g.firstProcessStep <= processStep.Summarize:

                """ Skip summarization for now
                changeBlinkRate(BLINK2)

                if args.summary == 0:
                    # summarize the transcript
                    summary = getSummary(transcript)

                    if args.savefiles:
                        f = open("history/" + timestr + "-summary" + ".txt", "w")
                        f.write(summary)
                        f.close()

                else:
                    # use the text file specified by the transcript argument
                    summaryFile = open(summaryArg, "r")
                    # read the summary file
                    summary = summaryFile.read()
                    logger.info("Using summary file: " + summaryArg)
                
                changeBlinkRate(BLINK_STOP)
                """


            # Keywords - set keywords
            if g.firstProcessStep <= processStep.Keywords:

                changeBlinkRate(BLINK3)

                if not g.isAudioKeywords:

                    if g.firstProcessStep < processStep.Keywords:
                        # extract the keywords from the summary
                        keywords = getAbstractForImageGen(transcript) 
                        logToFile.info("Keywords: " + keywords)

                        if g.isSaveFiles:
                            f = open("history/" + timestr + "-keywords" + ".txt", "w")
                            f.write(keywords)
                            f.close()

                    else:
                        # use the extract file specified by the extract argument
                        summaryFile = open(g.inputFileName, "r")
                        # read the summary file
                        keywords = summaryFile.read()
                        logger.info("Using abstract file: " + g.inputFileName)
                    
                else:
                    # use the transcript as the keywords
                    keywords = transcript

                changeBlinkRate(BLINK_STOP)

            # Image - set imageURL
            if g.firstProcessStep <= processStep.Image:

                changeBlinkRate(BLINK4)

                if g.firstProcessStep < processStep.Image:

                    # use the keywords to generate images
                    try:
                        imagesInfo = getImageURL(keywords)

                        imageURLs = imagesInfo[0]
                        imageModifiers = imagesInfo[1]

                        newFileName = postProcessImages(imageURLs, imageModifiers, keywords, timestr)

                        imageURLs = "file://" + os.getcwd() + "/" + newFileName
                        logger.debug("imageURL: " + imageURLs)

                        logToFile.info("Image file: " + newFileName)

                    except Exception as e:

                        print ("AI Image Error: " + str(e))
                        logToFile.info("AI Image Error: " + str(e))

                        newFileName = generateErrorImage(e, timestr)

                        imageURLs = "file://" + os.getcwd() + "/" + newFileName
                        logger.debug("Error Image Created: " + imageURLs)       
                
                else:
                    imageURLs = [g.inputFileName]
                    newFileName = g.inputFileName
                    logger.info("Using image file: " + g.inputFileName )

                changeBlinkRate(BLINK_STOP)
                
            # Display - display imageURL
            
            # display the image
            changeBlinkRate(BLINK_SLOW)
            logger.info("Displaying image...")

            # display the image with pillow
            #image = Image.open(newFileName)
            #image.show()

            display_image(newFileName, labelForImageDisplay)
            g_windowForImage.update_idletasks()
            g_windowForImage.update()
            
            changeBlinkRate(BLINK_STOP)

            # The end of the for loop
            changeBlinkRate(BLINK_STOP)
            # are we running the command line file args?
            if g.firstProcessStep > processStep.NoneSpecified:
                # We've done one loop to process a file and we're all done
                done = True
                time.sleep(20)   # persist the image display for 20 seconds
            else:
                #delay before the next for loop iteration
                if not g.isUsingHardwareButtons:
                    print("delaying " + str(g.loopDelay) + " seconds...")
                    time.sleep(g.loopDelay)

            # let the tkinter window events happen
            g_windowForImage.update_idletasks()
            g_windowForImage.update()
            
            # end of loop

    # all done
    if not g_isMacOS:
        # running on RPi
        # Stop the LED thread
        changeBlinkRate(BLINK_DIE)
        led_thread1.join()

        # Clean up the GPIO pins
        GPIO.cleanup()

    # exit the program
    print("\r\n")


logToFile.info("Starting Speech2Picture")

main()
exit()





