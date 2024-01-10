import os
from openai import OpenAI
from getVoices import *
from elevenlabs import *
from ObjectChar import *
import json
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
#from mergeAudio import *



client = OpenAI(api_key="YOUR KEY HERE")



background = """
    Limit the tokens for the response to 300 after the supplied beginning prompt.
	You are collaborativly creating an immersive play with the user.
	This story is intended to be read by children. Do not use characters that are not part of the initial prompt.
    Conclude each message with a request for open-ended input from the user, inviting them to actively contribute to the evolving storyline.
    Each paragraph must start with narrator if a character is not speaking.
	Ensure a seamless transition between lines, initiating each segment with the relevant character's name or the Narrator as appropriate.
	ABSOLUTELY ENSURE that each paragraph is in the following style "CHARACTER NAME: [CHARACTER DIALOGUE]."
	Maintain the integrity of the narrative by disregarding any extraneous responses that could disrupt the seamless flow of the unfolding tale.
    Should the user's input lack coherence or fail to align with the narrative possibilities, tactfully bypass the incongruity and proceed with the storyline.
    Keep the story to a minimum number of lines of dialogue and KEEP IT SHORT.
    """ 
    # Background to tell Chat how to respond and act in the story choose 1
"""
You are an amazing talented storyteller who will create a fabulous interactive story with the user. 
At the end of each message you send ask the user for open-ended input to continue the story you are creating. 
Ignore all irrelevant responses to the storyline to prevent breaking the immersion of your story. 
If the user's response does not make sense or if the user's response does not fit what could happen ignore what the user said. 
You will label each line with the name of the current character that is speaking or Narrator if the narrator is speaking. 
The beginning of each line starts with the name of the currently speaking character and narrator otherwise. 
""" # Another Background to tell Chat how to respond and act in the story choose 1

def startStory(history):
    '''Get initial story response from GPT.'''
    print("Starting story.")

    response = client.chat.completions.create(                          # Get response
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": background}]+history,   # Sends background to get start of story
        max_tokens=750,                                                 # Max length of message
    )

    response_text = response.choices[0].message.content
    #print(f"The response is \n{response_text}\n")                       # Testing what response was
    return ([{"role": "assistant", "content": response_text}],response_text)
    
def continueStory(history):
    '''Continue story by getting additional responses from GPT.'''
    userinput = input("Enter a message ")
    message = [{"role": "user", "content": userinput}]
    
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": background}] + history + message,    # Sends all messages sent in conversation to keep history
        max_tokens=300,                                                              # Max length of message
    )
    """
    #response_text = response.choices[0].message.content
    #print(f"The response is \n{response_text}\n")                                    # Testing what the response was
    #print(f"Used {response.usage.prompt_tokens} tokens with message {userinput} and {response.usage.total_tokens} in total")     # Testing to keep track of how many tokens used
    
    return ([{"role": "assistant", "content": response_text}] + message, response_text)

def generateStory(characters,sampleOutput):
    
    #print("Running.")
    # history = [{"role": "user", "content": "IMAGE_DESCRIPTION_FOR_START_OF_STORY"}] # History Records all previous messages from user and chat
    history = []    # Temporary
    #text = ""       # To store latest response from GPT
    text = sampleOutput
    
    response, text = startStory(history)        # Text is the raw response text

    readStory(characters, text, sampleOutput)
    '''
    history = history + response                            # Initialize story
    for i in range(5):
        response, text = continueStory(history=history)     # Get response 
        history = history + response                        # Build history
        print(f"The history is \n {history}")               # Print history
    '''

audioFiles = []
def readStory(characters, text,sampleOutput):
    charVoiceDict = {}
    # Store character voices in a dictionary instead of a list for faster retrieval
    for character in characters:
        charVoiceDict[character.name] = character.voice.name

    # Instantitate the narrator
    narrator = ObjectChar("Narrator","Woman","DoesItMatter?","WhyDoYouCare?","You'reAwfullyCurious","Ella - Narrator")
    
    ii = 0
    splitText = text.split("\n\n")                      # Split text into sections; list of sections
    for section in splitText:
        sectionComponents = section.partition(":")      # Partition section into names and dialogue (separated by colon)
        ''' Ideally GPT outputs 'Narrator:' as well so this section is not needed
        if (sectionComponents[2] == ""):                # In the case the partition never partitions anything
            currAudio = generate(
            text = text,                                # The text is just narrator text
            voice = narrator.voice                      # So set the voice to the narrator
            )
            play(currAudio)
            continue                                    # Skip rest of loop and move to next iteration
        '''
        currCharName = sectionComponents[0]             # Current character name
        currCharDialogue = sectionComponents[2]         # Current character dialogue
        if (currCharName == "Narrator" or currCharDialogue == ""):
            currAudio = generate(                       # Audio bytes
            text = currCharDialogue,                    # The text is just narrator text
            voice = narrator.voice                      # So set the voice to the narrator
            )
            #play(currAudio)

            ii+=1
            save(currAudio,f"audioFile{ii}.wav")        # Save audio bytes into .wav
            audioFiles.append(f"audioFile{ii}.wav")     # Add filename to audioFiles list
            continue
        currAudio = generate(
            text = currCharDialogue,                    
            voice = charVoiceDict[currCharName]         # Set the voice to the voice at the sepcified dictionary entry
        )
        #play(currAudio)

        ii+=1
        save(currAudio,f"audioFile{ii}.wav")            # Save audio bytes into .wav
        audioFiles.append(f"audioFile{ii}.wav")         # Add filename to audioFiles list

    for line in splitText:                              # Console text output
        print(line,"\n")
    # Fetch the service account key JSON file contents
    cred = credentials.Certificate('canvas-chronicles-firebase-adminsdk-hjxyc-6a9f80428e.json')
    # Initialize the app with a service account, granting admin privileges


    firebase_admin.initialize_app(cred, {
        'databaseURL': "https://canvas-chronicles-default-rtdb.firebaseio.com/"
    },'FIREBASE')
    ref = db.reference("/")
    jsonDict = {"Canvas2" : "filler"}
    jsonDict["Canvas2"] = sampleOutput
    # Serializing json
    json_object = json.dumps(jsonDict, indent=1)
 
    # Writing to sample.json
    with open("storyData.json", "w") as outfile:
      outfile.write(json_object)

    with open("storyData.json", "r") as f:
      file_contents = json.load(f)
    ref.set(file_contents)
      #combinedAudio = mergeAudio(audioFiles)

if __name__ == "__main__":
    generateStory()