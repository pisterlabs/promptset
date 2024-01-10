from bs4 import BeautifulSoup
import datetime, dotenv, itertools, requests
from gtts import gTTS
from gtts.tokenizer.pre_processors import abbreviations, end_of_line

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from moviepy import editor
from mutagen.mp3 import MP3

from os import getenv, path

import pathlib, pyttsx3, re
from PIL import Image

from string import punctuation
from sys import stderr
from typing import List

# Project imports
from exceptions.CustomException import CustomException
from IStory import IStory
from publishers.IPublisher import IPublisher
from utils.conclusion import conclusion
from utils.introduction import introduction
from utils.Utils import make_api_request, urlify, MULTISPACE

dotenv.load_dotenv()

class Story(IStory):
    id_obj = itertools.count(1)

    def __init__(self, progargs: dict) -> None:
        super().__init__()
        
        self.id = next(Story.id_obj)
        self.date = datetime.datetime.today().strftime('%Y-%m-%d')

        self.fb = progargs.get('fb')
        self.ig = progargs.get('ig')
        self.tw = progargs.get('tw')
        self.yt = progargs.get('yt')
        self.story_name = None
        
        self.url = progargs.get('url')
        self.mock = False
        self.text = {
            "Hindi": None,
            "English": None
        }
        self.title = {
            "Hindi": None,
            "English": None
        }
        self.sceneries = dict()
        self.llm = ChatOpenAI(temperature=0.4)

    def get_text(self):
        pattern = re.compile(r':\s$', re.MULTILINE)

        response = requests.get(self.url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        headings = soup.find_all('h1')
        paragraphs = soup.find_all('p')

        self.title["Hindi"] = [h.get_text() for h in headings][0].split(":")[0]
        self.text["Hindi"] = [re.sub(pattern, '-', p.get_text().replace("दु:खी", "दुखी").replace("छ:", "छह")) for p in paragraphs][0].split(":")[0]

        # Set the story name using the title
        self.story_name = self.title.get("Hindi").replace(" ", '').translate(str.maketrans('', '', punctuation))

    def translate(self) -> None:
        if self.text.get("Hindi"):
            text_to_translate = self.text.get("Hindi")

            # Set up the translation prompt, grammer (e.g. articles) omitted for brevity
            translation_template = '''
            Please act as a highly proficient translator for {from_lang} and {to_lang} languages.
            Between tags <TEXT> and </TEXT>, is the text of a popular story for kids in {from_lang} language. Please translate it to {to_lang} language.
            
            Return only the translation in {to_lang} language, not any text in original {from_lang} language.

            Ensure that returned text is highly engaging and suitable for a youtube channel higly popular amongst kids of 5 years to 14 years.

            <TEXT>{text}</TEXT>
            '''

            text_prompt = PromptTemplate(template=translation_template, input_variables=['from_lang', 'to_lang', 'text'])

            chain2 = LLMChain(llm=self.llm,prompt=text_prompt)

            # Extract the translated text from the API response
            input = {'from_lang': "Hindi", 'to_lang': "English", 'text': text_to_translate}
            translated_text = chain2.run(input)

            # Print the translated text
            self.text["English"] = translated_text
        else:
            raise CustomException("Please call .translate() after having fetched the Hindi story!")

    def get_sceneries(self):
        template_prompt = '''Please act as a highly creative and accomplished "visual illustrator" that extracts scenes from story. The story must be given within <STORY> and </STORY> tags, if not then please ask for story embeded in <STORY> and </STORY> tags. 
        Aesthetic scenes are elaborately explained and usually set in natural surroundings. Humans, trees, flowers, ornaments, water bodies, birds, mountains, valleys, Sun, Moon, stars, animals and other living beings are focused upon.
        Details about season, climate, weather, time of the day, colours and more are important.
        Scenes extracted from given story should not have names of characters, places or any other proper nouns. In a scene, no living being should be alone.
        Represented the extracted scenes as a Python dictionary where title of the scene is the key and value is a nested python dictionary with the detailed description of the scene as the first element. 
        The second element in the nested dictionary is a list of sentiments that can be used to explain the scenery in the detailed description. 
        Nothing else is required in the output Python dictionary.

        <STORY>{story}</STORY>
        '''

        sceneries_prompt = PromptTemplate(template=template_prompt, input_variables=['story'])

        chain2 = LLMChain(llm=self.llm,prompt=sceneries_prompt)
        input = {'story': self.text.get("English")}
        
        self.sceneries = chain2.run(input)
    
    def get_images(self, width: int = 512, height: int = 512, count: int = 1) -> None:        
        for key in self.sceneries:
            scenery_title = key
            scenery_prompt = '''Create a hyper-realistic scene described in these words: {description}. 
                                The lighting is cinematic and the photograph is ultra-detailed, with 8k resolution and sharp focus. 
                                The scene sentiments are explained by words such as {sentiments}.
                            '''.format(description=self.sceneries.get(key).get("description"), 
                                        sentiments=self.sceneries.get(key).get("sentiments"))

            img_data = None
            image_url = None
            output_path = None

            data = {
                'key': getenv('STABLEDIFFUSION_API_KEY_99'),
                'width': width,
                'height': height,
                'prompt': scenery_prompt,
                'negative_prompt': 'multiple images of the same scene',
                "enhance_prompt": "no",
                'guidance_scale': 8,
                "safety_checker": "no",
                'multi_lingual': 'no',
                'panorama': 'no',
                "samples": str(count)
            }

            headers = {
                'Content-type': 'application/json',
                'Accept': 'text/plain'
            }

            response = make_api_request(getenv('TEXT_TO_IMAGE_URL'), data, headers)
            if response.status_code == 200 and response.json().get('status').lower() != 'success':
                raise CustomException("Error: Image generation from Stable Diffusion did not work as expected!")

            # TODO: Use a for loop instead of 0 index
            # We use 0 for now as only one
            # image is requested by default
            image_url = response.json().get('output')[0]

            # TODO: Use make_api_request here?
            img_data = requests.get(image_url).content if image_url else None
            if img_data:
                img_name = './images/', urlify(scenery_title) + '.png'
                output_path = path.join(img_name)
                with open(output_path, 'wb') as handler:
                    handler.write(img_data)
            
    def get_audio(self, lib: str) -> str:
        final_text = introduction.get("Hindi") + "\n\n" + self.text.get("Hindi") + "\n\n" + conclusion.get("Hindi") + "\n\n"
        audio_path = path.join('./audios/', self.story_name + ".mp3")
        if lib.lower() == 'gtts':
            gttsLang = 'hi' # Hindi language

            replyObj = gTTS(text=final_text, lang=gttsLang, slow=True, pre_processor_funcs=[abbreviations, end_of_line])

            replyObj.save(audio_path)
        elif lib.lower() == 'pyttsx3':
            engine = pyttsx3.init()
            voices = filter(lambda v: v.gender == 'VoiceGenderFemale', engine.getProperty('voices'))
            for voice in enumerate(voices):
                print(voice[0], "Voice ID: ", voice[1].id, voice[1].languages[0])
                engine.setProperty('voice', voice[1].id)
                engine.say(final_text)
                engine.runAndWait()
        else:
            raise CustomException("Please use a valid speech processing library. {lib} is not valid!".format(lib=lib))
        
        return audio_path

    def get_video(self) -> str:
        video_name = self.story_name + ".mp4"
        audio_name = self.story_name + ".mp3"

        video_path = path.join('./videos', video_name)
        audio_path = path.join('./audios', audio_name)

        # Read the story audio mp3 file and set its length
        story_audio = MP3(audio_path)
        audio_length = round(story_audio.info.length) + 1
        
        # Glob the images and stitch them to get the gif
        path_images = pathlib.Path('./images/')
        # print("Path images: ", path_images.absolute())
        images = list(path_images.absolute().glob('*.png'))
        image_list = list()
        
        for image_name in images:
            image = Image.open(image_name).resize((800, 800), Image.Resampling.LANCZOS)
            image_list.append(image)
        
        print("Number of images: ", len(image_list)
                , "\nAudio length: ", audio_length)
        duration = int(audio_length / len(image_list)) * 1000

        # Creating the gif
        image_list[0].save(path.join('./videos/',"temp.gif"),save_all=True,append_images=image_list[1:],duration=duration)
        
        # Getting the vieo from the gif
        video = editor.VideoFileClip(path.join('./videos/', "temp.gif"))

        # Add audio to the video
        audio = editor.AudioFileClip(audio_path)
        video = video.set_audio(audio).set_fps(60)
        video.write_videofile(video_path)

        return video_path  
    
    def publish(self, publishers: List[IPublisher]) -> None:
        for publisher in publishers:
            publisher.login()

            publisher.publish()
            
            publisher.logout()