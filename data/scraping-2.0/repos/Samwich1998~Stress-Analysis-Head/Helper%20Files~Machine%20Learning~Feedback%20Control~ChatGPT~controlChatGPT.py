# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import os
import sys

# OpenAI
import openai

# Import Files for Machine Learning
sys.path.append(os.path.dirname(__file__) + "/Helper Files/")
import browserControl      # Methods for controlling the web browser.
import imageModifications  # Methods for working with and altering images.

# -------------------------------------------------------------------------- #
# ---------------------------- ChatGPT Interface --------------------------- #

class chatGPTController:
    def __init__(self, userName = ""):
        # General model parameters.
        self.textEngine = "gpt-4-0613" # See text models at https://platform.openai.com/docs/models/gpt-4
        self.userName =userName        # A unique username for the client. Not used for personalization.
        
        # Set up the OpenAI API client
        openai.api_key = "".join("s k - k s OqP6JXLjX A Mqmi2G JlT3Blbk FJltwvFxs YE4A Mw8KA Lpu6".split(" "))
        
        # Instantiate necesarry classes.
        self.browserController = browserControl.browserControl()
    
    # ---------------------------------------------------------------------- #
    # -------------------------- General Methods --------------------------- #
    
    def displayImage(self, response):
        # Get the image URL.
        image_url = self.getImageURL(response)
        # Open the image URL with the webdriver.
        self.browserController.open_url(image_url)
    
    def printModels(self):
        print(openai.Model.list())
        
    def getImageURL(self, response):
        return response['data'][0]['url']
        
    # ---------------------------------------------------------------------- #
    # ---------------------------- Text Methods ---------------------------- #
        
    def getTextReponse(self, textPrompt):
        # Generate a response
        completion = openai.Completion.create(
            engine=self.textEngine,
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )

        response = completion.choices[0].text
        
        return response
    
    # ---------------------------------------------------------------------- #
    # --------------------------- Image Methods ---------------------------- #
    
    def getImageResponse(self, textPrompt):
        # Assert the proper data format.
        assert len(textPrompt) <= 1000, f"The maximum length is 1000 characters for text. Given {len(textPrompt)} characters"
        assert isinstance(textPrompt, str), f"Expecting the text prompt to be a string. Given type {type(textPrompt)}. Value: {textPrompt}"
        
        # Interface with chatGPT API.
        response = openai.Image.create(
            response_format = "url",  # The format in which the generated images are returned. Must be one of url or b64_json.
            prompt=textPrompt,        # A text description of the desired image(s). The maximum length is 1000 characters.
            size="1024x1024",         # The size of the generated images. Must be one of 256x256, 512x512, or 1024x1024.
            user = self.userName,     # A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. Learn more.
            n=1,                      # The number of images to generate. Must be between 1 and 10.
        )
        
        return response
    
    def varyImageResponse(self, image, mask, textPrompt):
        response = openai.Image.create_edit(
            response_format = "url",  # The format in which the generated images are returned. Must be one of url or b64_json.
            user = self.userName,     # A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. Learn more.
            prompt=textPrompt,        # A text description of the desired image(s). The maximum length is 1000 characters.
            size="1024x1024",         # The size of the generated images. Must be one of 256x256, 512x512, or 1024x1024.
            image=image,              # The image to edit. Must be a valid PNG file, less than 4MB, and square. If mask is not provided, image must have transparency, which will be used as the mask.
            mask=mask,                # An additional image whose fully transparent areas (e.g. where alpha is zero) indicate where image should be edited. Must be a valid PNG file, less than 4MB, and have the same dimensions as image.
            n=1,                      # The number of images to generate. Must be between 1 and 10.
        )
        
        return response
    
    # ---------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Instantiate class.
    gptController = chatGPTController()
    imageController = imageModifications.imageModifications(os.path.dirname(__file__) + "/_savedImages/")
    
    prompts = [
        "I have a state anxiety score of 60 out of 80, a postive affectivity score of 10 out of 25, and a negative affectivity score of 18. Generate an image that will reduce my anxiety GIVEN the anxiety scores I have told you. For example, you can display a lovely mountain range that is peaceful and transquil, use your judgement.",
        "After your last image, my STAI state anxiety (20 - 80) score went from 60 to 80 out of 80, my postive affectivity score went from 10 to 15 out of 25, and my negative affectivity score went from 18 to 20 out of 25. Generate an image that will reduce my anxiety GIVEN the anxiety scores I have told you. For example, you can display a lovely mountain range that is peaceful and transquil, use your judgement.",
        "After your last image, my STAI state anxiety (20 - 80) score went from 80 to 50 out of 80, my postive affectivity score went from 15 to 14 out of 25, and my negative affectivity score went from 20 to 15 out of 25. Generate an image that will reduce my anxiety GIVEN the anxiety scores I have told you. For example, you can display a lovely mountain range that is peaceful and transquil, use your judgement.",
    ]
    
    prompts = [
        # "Generate a calming image of a realistic beautiful beach.",
        # "Display a calming image of a realistic outdoor view of a snowy oasis on christmas night.",
        "Display a calming image of a realistic indoor view of a japenese zen house with a firepit, a koi pond, and the jungle.",
    ]
    
    # Flags for which programs to run.
    displayPromptedImages = False
    editPromptedImage = True
    
    # ---------------------------------------------------------------------- #
    # --------------------- Generate Images for Display -------------------- #
    
    if displayPromptedImages:
        # For each prompt.
        for prompt in prompts:
            # Ask chatGPT to generate an image response.
            response = gptController.getImageResponse(prompt)
            gptController.displayImage(response)
            
    # ---------------------------------------------------------------------- #
    # ------------------------ Edit a Prompted Image ----------------------- #
    
    if editPromptedImage:        
        # Ask chatGPT to generate an image response.
        initialResponse = gptController.getImageResponse(prompts[0])
        gptController.displayImage(initialResponse)
        
        # Get the image content from the URL.
        image_url = gptController.getImageURL(initialResponse)
        imageRGBA = imageController.pullDownWebImage(image_url) # Convert the the chatGPT image format.
        
        # Make a mask for the image.
        imageMaskRGBA = imageController.make_top_half_translucent(imageRGBA)
        # imageMaskRGBA = imageController.remove_hex_color(imageRGBA, "#FFFFFF")
        # imageMaskRGBA = imageController.remove_similar_colors(imageRGBA, "#FFFFFF", tolerance = 250)
        
        # Conver the images into the correct chatGPT format.
        imageMask = imageController.rbga2ByteArray(imageMaskRGBA)
        imageByteArray = imageController.rbga2ByteArray(imageRGBA)

        # Regenerate the image with the mask filled in.
        finalResponse = gptController.varyImageResponse(imageByteArray, imageMask, prompts[0])
        gptController.displayImage(finalResponse)






