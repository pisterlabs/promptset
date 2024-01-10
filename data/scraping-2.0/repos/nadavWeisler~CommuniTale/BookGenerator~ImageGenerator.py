import openai
import os


class ImageGenerator:
    # static var to get a unique image name - should be in Shlomi's code
    image_counter = 0

    def __init__(self):
        self.image_url = ""
        self.API_K = os.getenv("GPT_API_KEY")
        openai.api_key = self.API_K


    def getImageFromPrompt(self, prompt: str):
        """
        Main entry point for ImageGenerator, will get a prompt for DALL-E and should generate an
        Image based on
        that prompt (format of image to be determined)
        :param prompt:
        :return:
        """
        print(f"Got request for Image from prompt: {prompt}")
        ##########################################################
        # Request To DALL-E

        # *** requires: ***
        # prompt(str)

        # *** optional: ***
        # n(int: default 1),
        # size(str: default 1024x1024),
        # response format(str: default to url)
        # user(str: end-user)
        ##########################################################

        # the rest are default, for now set to this - eventually I will get it in prompt:
        img_size = "1024x1024"

        dall_e_response = openai.Image.create(api_key=self.API_K, prompt=prompt, size=img_size)
        self.image_url = dall_e_response['data'][0]['url']
        return self.image_url


# if __name__ == "__main__":
#     # Instantiate the class
#     imageGen = ImageGenerator()
#
#     # Generate images:
#     imageGen.getImageFromPrompt(
#         "A drawing of a group of university students creating a children's book"
#     )
#
#     # Download the images:
#     imageGen.convert_url_to_png()
