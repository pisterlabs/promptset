import random
import sys
import time
import requests
from PIL import Image
from io import BytesIO
import base64

#pip install openai


def main():

    def secret_code_function():
        """
        :param name: secret_code_function
        :type secret_code_function: str
        :returns: the encoded answer
        :rtype: str
        """
        def encode(key, clear):
            """
            :param name: key
            :type key: str
            :returns: result after being encoded
            :rtype: str
            """
            enc = []
            for i in range(len(clear)):
                key_c = key[i % len(key)]
                enc_c = chr((ord(clear[i]) + ord(key_c)) % 256)
                enc.append(enc_c)
            return base64.urlsafe_b64encode("".join(enc).encode()).decode()

        def decode(key, enc):
            """
            :param name: key
            :type key: str
            :returns: result after being encoded
            :rtype: str
            """
            dec = []
            enc = base64.urlsafe_b64decode(enc).decode()
            for i in range(len(enc)):
                key_c = key[i % len(key)]
                dec_c = chr((256 + ord(enc[i]) - ord(key_c)) % 256)
                dec.append(dec_c)
            return "".join(dec)

        secret_openai_jumble = "w6HDkMKZw63CscK1w4PCnMODw5nCs8OlwrbCusOBw6bDgMOaw5rDlcOCw6TCssOCwp_Cq8OZw5DDkMKywr3DicKmw6bDncKdw6HDkcOew4nCusOew4bCscKewq_Dn8K_w6TDmsOS"
        print("Enter the secret code! (Hint: professor's full name, no spaces, no caps)")
        secret_code = input().lower()
        openai_api_key = decode(secret_code, secret_openai_jumble)
        return openai_api_key

    openai_api_key = secret_code_function()

    print("Welcome to the quick sketch practice game! Answer yes to the following questions to randomize a character description! Once the character description is randomized, you will have 30 seconds to draw this character on a piece of paper or sketchbook. After time is up, I'll draw my own sketch and you can see if yours looks like mine!. If you are playing with a friend, the sketch that looks more like mine wins! If you are playing alone, you can use this prompt as inspiration, or see how fast you can sketch the prompt, then see if your drawing looks similar to mine.")

    print("Would you like to start your character?")
    
    artist = input()
    #def start(artist):
    if (artist == "yes"):
            """
            :param artist
            :type artist: str
            :returns: print function
            :rtype: str
            """
            ##two lines below make case sensitive?
            #artist=input(answer_yes)
            #answer_yes = answer_yes.casefold()
        #return
            print("Great! Let's get Started!")
    if (artist == "no"):
        print ("well you just suck!")
        sys.exit()
        
    
    print("Would you like to randomize your type of character?")
    
    
    def character_type(artist):
        """
        :param artist
        :type artist: str
        :return: type
        :rtype: str
        """
    artist = input()
    if (artist == "yes"):
            ##two lines below make case sensitive?
            #artist=input(answer_yes)
            #answer_yes = answer_yes.casefold()
            #return character_type
        character_type = ["Human", "Animal"]
        idx = random.randrange(2)
        print("Your character is a" , character_type[idx])
    #character_type[idx]=type
    if (character_type[idx] == "Animal"):
    #elif(idx == "Animal"):
        file_path = "animal.txt"
        #try:
        with open(file_path, 'r', encoding='utf-8-sig') as file:
                lines = file.readlines()
                if lines:
                    random_line = random.choice(lines)
                    animal_type = random_line.strip()
                    print("a(an)", animal_type)
    
    
    #print("Would you like to choose the gender?")   
    print("Would you like to know the gender?")
    artist = input()
    if (artist == "yes"):
        gender = ["Male","Female"]
        random_gender = random.choice(gender)
        print("Your character is",random_gender) 
    #if (artist == "yes"):
        #print("Type your gender choice")
        #artist = input()
        #type_gender = f" {artist}"

        
    
    print("Would you like to know the vibe of your character?")


    artist = input()
    file_path = "emotion.txt"
    with open(file_path, 'r', encoding='utf-8-sig') as file:
            lines = file.readlines()
            if lines:
                random_line = random.choice(lines)
                emotion = random_line.strip()
                

    file_path = "theme.txt"
    with open(file_path,'r', encoding='utf-8-sig') as file:
                lines = file.readlines()
                if lines:
                    random_line = random.choice(lines)
                    theme = random_line.strip()

                print("Your character is a(an)",emotion, theme)
    



        
    #open.api_key = os.getenv("OPENAI_API_KEY")
    if (character_type[idx] == "Animal"):
        user_prompt = (character_type[idx],animal_type,random_gender,emotion,theme)
    if (character_type[idx]) == "Human":
        user_prompt = (character_type[idx],random_gender,emotion,theme)


    
    user_prompt_string = " ".join(user_prompt) 
    openai_prompt_string = user_prompt_string + " pen sketch"

    def draw_sketch():   
        """
        :param name:openai_api_key
        :type openai_api_key:str
        :returns: URL for ai generated image
        :rtype: str
        """ 
        from openai import OpenAI
        client = OpenAI(api_key=openai_api_key)

        response = client.images.generate(
        model="dall-e-2",
        prompt=openai_prompt_string,
        size="512x512",
        quality="standard",
        n=1,
        )
        image_url = response.data[0].url


    

        # URL of the image
        image_url = image_url

        # Send a GET request to the URL to fetch the image
        response = requests.get(image_url)


        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Read the image content from the response
            image_data = response.content
            
            # Create a PIL Image object from the image data
            image = Image.open(BytesIO(image_data))
            
            # Display the image
            image.show()
        else:
            print("Failed to fetch the image. Status code:", response.status_code)


    print("\n"*50)
    #print(f"Draw this! {user_prompt_string}")
    if (character_type[idx] == "Animal"):
        print("In sum, your character is a(an)",character_type[idx],animal_type,random_gender,"who is a(an)",emotion,theme,"!")
    if (character_type[idx]) == "Human":
        print("In sum, your character is a(an)",character_type[idx],random_gender,"who is a(an)",emotion,theme,"!")
    print(f"Race to draw the newly generated prompt! You have 30 seconds!")

    seconds = 30

    while seconds > 0:
        print(f"Time left: {seconds} seconds", end='\r')  # Use '\r' to overwrite the previous line
        time.sleep(1)  # Pause for one second
        seconds -= 1
    print("PENCILS DOWN! "*1000)
    print("\nMy turn to draw, one second...")
   
    try:
        draw_sketch()
    except:
        print("Oops, You may have put in the wrong password...Or you may need to uppdate to openAi version 1.2")
    print("Was your drawing close to mine?")


if __name__ == "__main__":
   main()
