#A free, and open source game that allows you to go through infinite possibilities, I present to you, 
#Infinity Worlds

#As always, free as well as open source!

#Standard libraries:
import os
import time
import sys
import colors
import logos
import random

#Import config parser:
from configParser import createConfigFile
from configParser import importConfig
from configParser import getHomeDirectory
from configParser import convertStringToCallableClass
#Used for the AI (Uses GPT))
import openai

#Initialize the config file:
createConfigFile()

#Defining the main template to use:
mainTemplate = importConfig("IWSettings", os.path.join(getHomeDirectory(), ".IWconfig.json"), isSettings = True)

#Define hyperparameters (while passing downt he mainTemplate above):
openAIAPIKey, gptModel, skipIntro, initialInstructions, disableAnimations, disableActivityRecaps, animationSpeed, UITextColor, UITextStyle, StoryTextColor, StoryTextStyle, disableLogo, LogoColor, continueFlag, inputFlag, quitCommand, storySavePath = importConfig(mainTemplate, os.path.join(getHomeDirectory(), '.IWconfig.json'))

#Convert the animation speed to float:
animationSpeed = float(animationSpeed)

openai.api_key = openAIAPIKey
#Clearscreen function (Visual purposes):
def clearscreen():
	if(os.name == 'posix'):
		os.system('clear')
	else:
		os.system('cls')
		
def delay_print(s, t, disableAnimations = False):
	if disableAnimations == False:
		for c in s:
			sys.stdout.write(c)
			sys.stdout.flush()
			time.sleep(t)
	else:
		print(s)
			


def intro(skipIntro = False, disableAnimations = False):
	
	if skipIntro == False:
		clearscreen()
		delay_print("Welcome to the World of Infinite Possibilities...", animationSpeed, disableAnimations = disableAnimations)
		time.sleep(2)
		print("\n")
		delay_print("What shall the next story be?", animationSpeed, disableAnimations = disableAnimations)
		print("\n")

#Get initial color/style attributes:

UITextColorClass = convertStringToCallableClass(UITextColor)
UITextStyleClass = convertStringToCallableClass(UITextStyle)
StoryTextColorClass =  convertStringToCallableClass(StoryTextColor)
StoryTextStyleClass =  convertStringToCallableClass(StoryTextStyle)

LogoColorClass =  convertStringToCallableClass(LogoColor)

def printColor(textTypeColor, textTypeStyle):
	#Resets the color:
	print(colors.style.reset)
	
	#Prints the text style:
	print(textTypeStyle)
	#Prints the color:
	
	print(textTypeColor)
	
def printLogo(LogoColor, renderSpeed, disableLogo = False, disableAnimations = False):
	if disableLogo != True:
		#Define the Logo:
		randomLogo = convertStringToCallableClass(f"logos.Logos.logo{str(random.randint(1, 8))}")
		print(LogoColor)
		delay_print(randomLogo, renderSpeed, disableAnimations = disableAnimations)

def saveStoryToFile(promptcontents, savePath):
	currentTime = time.ctime()
	saveFileName = f"{currentTime}__IWStory.txt"
	f = open(os.path.join(savePath, saveFileName), 'w')
	f.writelines(promptcontents)
	f.close()

printColor(UITextColorClass, UITextStyleClass)
#Intro
intro(skipIntro = skipIntro, disableAnimations = disableAnimations)

#Generate the text with the help of GPT.
def generate(prompt):
	contents = openai.ChatCompletion.create(
		model=gptModel,
		messages=[
			{
				"role": "user",
				"content": str(prompt)
			}
		],
		temperature=1,
		max_tokens=256,
		top_p=1,
		frequency_penalty=0,
		presence_penalty=0
	)
	
	my_openai_obj = list(contents.choices)[0]
	generated = my_openai_obj.to_dict()['message']['content']
	
	return str(generated)

#The MainMenu/MainACTIVITY:
while True:
	#Clear the screen.
	time.sleep(1.5)
	clearscreen()

	#Print the logo if it is not disabled:
	
	printLogo(LogoColorClass, .003, disableLogo = disableLogo, disableAnimations = disableAnimations)

	printColor(UITextColorClass, UITextStyleClass)
	questions = "\n\n1) Start\n2) Credits\n00) Exit"
	delay_print(questions, animationSpeed, disableAnimations = disableAnimations)
	MainMenu = int(input(f"\n{inputFlag} "))

	if MainMenu == 1:
		clearscreen()
		#The contents:
		promptcontents = []

		#Print Logo:
		printLogo(LogoColorClass, .001, disableLogo = disableLogo, disableAnimations = disableAnimations)

		#Print initial question:
		printColor(UITextColorClass, UITextStyleClass)
		
		delay_print(f"\n{initialInstructions}", animationSpeed, disableAnimations = disableAnimations)
		command = str(input(f"\n{inputFlag} "))
		promptcontents.append(f"\n{command}\n")
		#Feed the AI:
		printColor(StoryTextColorClass, StoryTextStyleClass)
		outputtext = generate("Write an adventurous two sentences about being " + command + " in second person point of view")
		delay_print(outputtext, animationSpeed, disableAnimations = disableAnimations)
		#Append what happened.
		promptcontents.append(f"\n{outputtext}\n")
		#While True, Continue the story:
		i = 1
		while True:
			promptcontentscurated = ' '.join(promptcontents)
			
			#Summarize to reduce token amount in API request:
			if i == 1:
				summary = generate("Summarize the content " + command + " by getting the names and the traits of each character accordingly, in one to two simple sentences")
			else:
				summary = summarycontinue
			printColor(UITextColorClass, UITextStyleClass)
			#What happens next in the story:
			whatnext = input(f"\n\n{continueFlag} {inputFlag} ")

			promptcontents.append(f"\n{whatnext}\n")
			if quitCommand in whatnext:
				#If a directory is provided in the config file:
				if storySavePath != "":
					saveToFile = input("\nSave story to file? ( (1) Yes / (2) No ): ")

					if int(saveToFile) == 1:
						saved = False
						while saved == False:
							try:
								saveStoryToFile(promptcontents, storySavePath)
								saved = True
							except:
								storySavePath = input("\nERR: Error saving file at destination provided. Provide a new location: ")
				delay_print("\nExiting the story due to an unexpected interrupt...", animationSpeed, disableAnimations = disableAnimations)
				break
			#Prompt to give to AI (Add the summary, and what happens next:)
			continuationstory = generate("Continue what happens next based on the information of: " + summary + " " + whatnext + " in three sentences using second person point of view")
			
			summarycontinue = generate("Summarize the content " + continuationstory + " by getting the names and the traits of each character accordingly, in one to two simple sentences")
			#Append the new content so the algorithm knows what happened in the previous details added.
			promptcontents.append(f"\n{continuationstory}\n")
			#Print what happens next:

			#Change certain things to second person:
			printColor(StoryTextColorClass, StoryTextStyleClass)

			if disableActivityRecaps == False:
				displaywhatnext = generate("Convert this sentence to second person: " + whatnext)
				delay_print("Activity: "+ displaywhatnext, animationSpeed)
			delay_print("\n\n" + continuationstory + "\n", animationSpeed, disableAnimations = disableAnimations)
			i += 1
	elif MainMenu == 2:
		clearscreen()
		print("\nInfinityWorlds brings you a configurable and fully customizable experience to enjoy creating limitless amounts of stories. This is a fun project I wanted to make. I always wanted to make one of these AI text adventures, and now with InfinityWorlds V2, the experience is just even better :)\n\nCreated with passion by PatzEdi, and of course, OpenAI and their GPT models (HUGE credits to the Chat GPT team over at OpenAI!)")

		input("\nPress ENTER to go back to the Main Menu: ")
	elif MainMenu == 0:
		delay_print("\nYou are leaving the worlds...", animationSpeed, disableAnimations = disableAnimations)
		exit()
	else:
		print("\nNot a valid option!")
		delay_print("\nGoing back...", animationSpeed, disableAnimations = disableAnimations)
