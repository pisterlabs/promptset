""" 
The main meat of the app that generates a user profile and social media content using OpenAI's APIs.
"""

import json
import os
from random import choice
import openai
import requests

from lambchop.utils import get_config_options, printc, save_to_file

OUT_DIR = get_config_options()[0]


def main(country="USA", language="English", style="casual", extras=True):
    """
    Main function for generating a user profile and social media content.
    """

    # Generate and output user profile
    user_profile = UserProfile(country=country, language=language, style=style)
    user_profile.generate_profile()
    saved_profile_path = user_profile.output_profile(OUT_DIR)

    # Generate and save social media avatar
    image_generator = ImageGenerator(user_profile)
    image_generator.generate_image()
    image_generator.save_image()

    if extras:
        extra_stuff = ExtraStuff(saved_profile_path)
        extra_stuff.suggest_subreddits()
        extra_stuff.create_twitter_post()


def load_profile_data(profile_path):
    """
    Load AI generated profile data from its JSON home.

    If the file is not found,the user is prompted to generate a new profile.
    If the user agrees, a new profile is generated, saved to the specified output directory, and its data is returned.
    If the user declines, a FileNotFoundError is raised.
    """

    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        printc(f"[!] Could not find the specified profile JSON file: {profile_path}")
        selection = input("Would you like to generate a new profile? [Y/n] ").strip().lower()
        if selection in ["", "y", "yes"]:
            profile = UserProfile()
            profile.generate_profile()
            profile_path = profile.output_profile(OUT_DIR)
            with open(profile_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            printc("[!] Ok then, nothing for me to do here. Exiting...")


class UserProfile:
    """
    Class that handles generating profile content using OpenAI's ChatGPT API.
    """

    def __init__(self, country="", language="", style=""):
        self.language = language
        self.style = style
        self.country = country
        self.profile_info = {
            "full_name": "",
            "age": "",
            "country": "",
            "city": "",
            "bio": "",
            "tagline": "",
            "username": "",
            "avatar": ""
        }

    def generate_profile(self):
        """
        This method utilizes the OpenAI ChatGPT API to generate a realistic user profile. The profile is tailored
        based on specified parameters like language, age range, country, occupation, and writing style. It constructs
        a prompt that includes these parameters and instructs the model to generate a profile adhering to the
        provided schema.

        The method communicates with the ChatGPT model to create a user profile according to the prompt and schema.
        The response is then parsed and converted into a dictionary representing the profile information.
        This dictionary is stored within the class instance for further processing or output.

        To ensure accuracy and relevance, the method utilizes predefined lists of age ranges and occupation types.
        The prompt is structured to guide the model in generating a profile bio that reflects the specified attributes
        while adhering to the chosen language and writing style.

        This method plays a crucial role in generating realistic user profiles that can be used for various purposes,
        such as generating content, testing, or populating databases.
        """

        # Added these lists because GPT was just constantly making 32-year-old tech bros
        age_range = ["18-25", "26-32", "33-40", "41-45"]
        occupation_types = ["techie", "artist", "entrepreneur", "student", "freelancer", "teacher", "volunteer",
                            "engineer", "writer", "musician", "lawyer", "doctor", "scientist", "researcher",
                            "government worker", "salesperson", "retail worker", "customer service rep"]
        selected_age_range = choice(age_range)
        selected_occupation = choice(occupation_types)

        prompt = f"""Create a realistic user profile in {self.language} for someone aged {selected_age_range}
        from {self.country}. This person works as a {selected_occupation}. 
        The bio should reflect their age, country, and occupation. 
        If English is chosen and the person isn't a native speaker, 
        the bio should reflect typical non-native English from that country. 
        Write in a {self.style} style. The user's last name shouldn't be Doe."""

        printc("[*] Generating basic bio")
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Respond with the requested info ONLY, do not include the prompt or "
                                              "preface in the response and only provide a RFC8259 compliant JSON "
                                              "response following this format without deviation."},
                {"role": "user",
                 "content": f"{prompt} Create the profile based on the following schema: {self.profile_info}. The "
                            "'avatar' value should be a short simple prompt to generate a social media avatar for "
                            "the user with AI art following the format 'Stylized social media avatar of '. "
                            "Lastly, be creative and unique with the value of 'username'."}
            ]
        )

        # Parse the response into a dictionary
        profile_data = completion.choices[0]["message"]["content"]

        # Convert the string to a dictionary using JSON
        parsed_profile = json.loads(profile_data)
        self.profile_info = parsed_profile

    def convert_to_json(self):
        """
        Converts the profile info dictionary to a JSON-formatted string.
        """

        json_str = json.dumps(self.profile_info, indent=4)
        return json_str

    def output_profile(self, out_dir):
        """
        Writes the profile info to a JSON file and returns the path to the saved file.
        """

        return save_to_file(self.convert_to_json(), self.profile_info["full_name"], out_dir)


class ImageGenerator:
    """
    Class that handles generating a social media avatar for the user profile using OpenAI's DALL-E API.
    """

    def __init__(self, profile):
        self.profile = profile
        self.image_url = None

    def generate_image(self):
        """
        Generates a social media avatar based on the profile bio.
        """

        printc("[*] Generating social media avatar")
        prompt = f"{self.profile.profile_info['avatar']}"

        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="512x512"
        )

        self.image_url = response["data"][0]["url"]

    def save_image(self):
        """
        Saves the generated image to the specified directory.
        """

        os.makedirs(OUT_DIR, exist_ok=True)

        filename = self.profile.profile_info["full_name"].lower().replace(" ", "_") + ".png"
        output_file_path = os.path.join(OUT_DIR, filename)

        # Download the image and save it to the specified directory
        response = requests.get(self.image_url, timeout=10)
        with open(output_file_path, "wb") as f:
            f.write(response.content)


class ExtraStuff:
    """
    Initialize the ExtraStuff instance.

    This constructor initializes an instance of the ExtraStuff class. It takes the path to a profile JSON file
    and an optional output directory. The profile data is loaded from the JSON file to be used in later methods.
    """

    def __init__(self, profile_path):
        self.profile_path = profile_path
        self.output_dir = OUT_DIR
        with open(self.profile_path, "r", encoding="utf-8") as f:
            self.profile_data = json.load(f)

    def suggest_subreddits(self):
        """
        Suggests subreddits for the generated profile to follow based on the bio.

        Utilizes the OpenAI ChatCompletion API to suggest subreddits for the generated user profile.
        It constructs a prompt using the profile's bio and communicates with the model to generate subreddit
        suggestions based on the provided bio. The resulting suggestions are saved to a text file in the output
        directory.
        """

        printc("[*] Suggesting subreddits for the generated profile")
        bio = self.profile_data.get("bio", "")

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are chatting with a user who needs subreddit suggestions based on a bio."},
                {"role": "user", "content": f"Suggest subreddits for someone with this bio: {bio}"}
            ]
        )

        result = completion.choices[0]["message"]["content"].strip().split("\n")
        save_to_file(result, f"{self.profile_data['full_name']}_subreddits", self.output_dir, extension="txt")

    def create_twitter_post(self):
        """
        This method utilizes the OpenAI ChatCompletion API to create a Twitter post in character for the generated
        user profile. It constructs a prompt using the profile's bio and communicates with the model to generate
        a tweet that captures the personality and traits of the profile. The resulting tweet is saved to a text
        file in the output directory.
        """

        printc("[*] Writing a tweet for the generated user")
        bio = self.profile_data.get("bio", "")

        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system",
                 "content": "You are chatting with a user who needs you to write a short Twitter tweet."},
                {"role": "user",
                 "content": f"Write something that would be in character for someone with this bio: {bio}. "
                            "Don't just rephrase what's in the bio, simply use it "
                            "as a basis for understanding their personality. "
                            "Do not use any hashtags or links in the tweet."}
            ]
        )

        result = completion.choices[0]["message"]["content"].strip().split("\n")
        save_to_file(result, f"{self.profile_data['full_name']}_tweet", self.output_dir, extension="txt")


if __name__ == "__main__":
    main()
