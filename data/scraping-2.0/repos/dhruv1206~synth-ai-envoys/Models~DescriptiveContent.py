import json

import openai

from Models.OriginalPressRelease import OriginalPressRelease
from consts import API_KEY, PrStatus

from utils import count_tokens


class DescriptiveContent:
    def __init__(self, pr_id, title, descriptive_text, image_urls, key_words, date, language, video_url=None,
                 audio_urls=[],
                 status=PrStatus.PENDING.value, ministry=""):
        self.status = status
        self.prId = pr_id
        self.title = title
        self.descriptive_text = descriptive_text
        self.imageUrls = image_urls
        self.key_words = key_words
        self.date = date
        self.language = language
        self.videoUrl = video_url
        self.ministry = ministry
        self.audioUrls = audio_urls

    def to_json(self):
        return {
            "prId": self.prId,
            "title": self.title,
            "description": self.descriptive_text,
            "imageUrls": self.imageUrls,
            "keyWords": self.key_words,
            "date": self.date,
            "language": self.language,
            "videoUrl": self.videoUrl,
            "status": self.status,
            "ministry": self.ministry,
            "audioUrls": self.audioUrls
        }

    @classmethod
    def from_json(cls, json_dict):
        return cls(
            json_dict["prId"],
            json_dict["title"],
            json_dict["description"],
            json_dict["imageUrls"],
            json_dict["keyWords"],
            json_dict["date"],
            json_dict["language"],
            json_dict.get("videoUrl"),
            json_dict.get("status"),
            json_dict.get("ministry"),
            json_dict.get("audioUrls")
        )


class DescriptiveContentGenerator:
    def __init__(self):
        openai.api_key = API_KEY
        self.engine = "text-davinci-003"  # Choose an appropriate engine
        self.temperature = 0.7
        self.max_tokens = 4096  # You can adjust this based on the desired length

    def generate_descriptive_content(self, original_press_release) -> DescriptiveContent:
        print(f"Original Press Release:  {original_press_release.to_json()}")
        prompt = f'''Given the press release: "{original_press_release.content}", generate more descriptive and 
        summarized content (such that whole context of the news is clear), including visuals and scene descriptions. 
        Consider '\\' as a line break. The format of the result should be in json and the fields should be:

- descriptive text which should be a list of paragraphs (each paragraph should be a string) as i need to display them 
on separate slides, each item in list should be generate so that it can be fitted in single slide - key words/phrase 
should be a list of words/phrase, the key words/phrase should be ultra relevant such that searching for the key 
words/phrase should give relevant images for each slide. - the descriptive text should not be very long but a 
summary, and should be written in a way that gTTS can convert it to speech nicely, ie. grammatically correct and also 
punctuation should be perfect. - the sum of all characters in descriptive_text should not exceed 2000 characters. 
result should be something like : {{
        
        "descriptive_text": [
            "The quick brown fox jumps over the lazy dog.",
            "The quick brown fox jumps over the lazy dog."
        ],
        "key_words": [
            "dog",
            "indian flag"
        ],
        "language": "english"
    }}

    The result should only contain the above fields and nothing else. The result should be in json format
        '''

        response = openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens - count_tokens(prompt),
        )

        response = json.loads(response.choices[0].text)

        pr_id = original_press_release.prId
        title = original_press_release.title
        descriptive_text = response["descriptive_text"]
        image_urls = original_press_release.imageUrls
        key_words = response["key_words"]
        date = original_press_release.date
        language = response["language"]

        content = DescriptiveContent(pr_id, title, descriptive_text, image_urls, key_words, date, language)
        return content


if __name__ == "__main__":
    descriptiveContentGenerator = DescriptiveContentGenerator()
    dummyOriginalPressRelease = OriginalPressRelease("1",
                                                     """President of India to confer National Teachers’ Award 2023 to 
                                                     75 selected teachers on 5th September 2023""", """Hon’ble 
                                                     President of India Smt Droupadi Murmu shall confer the National 
                                                     Teachers’ Award 2023 to 75 selected Awardees on 5th September 
                                                     2023 at Vigyan Bhawan, New Delhi. Every year, India celebrates 
                                                     5th September, the birth anniversary of Dr. Sarvepalli 
                                                     Radhakrishnan, as National Teachers’ Day. The purpose of 
                                                     National Teachers’ Award is to celebrate the unique contribution 
                                                     of teachers in the country and to honor those teachers who, 
                                                     through their commitment and dedication, have not only improved 
                                                     the quality of education but also enriched the lives of their 
                                                     students. Each award carries a certificate of merit, 
                                                     a cash award of Rs. 50,000 and a silver medal. The awardees 
                                                     would also get an opportunity to interact with Hon’ble Prime 
                                                     Minister.

Department of School Education & Literacy, Ministry of Education has been organising a National level function on 
Teachers Day every year to confer the National Awards to the best teachers of the country, selected through a 
rigorous, transparent selection process. From this year, the ambit of National Teachers’ Award has been expanded to 
include teachers of Department of Higher Education and Ministry of Skill Development & Entrepreneurship. 50 School 
Teachers, 13 teachers from Higher education and 12 teachers from Ministry of Skill Development & Entrepreneurship 
will be awarded this year.

With a view to recognize innovative teaching, research, community outreach and novelty of work the nominations were 
sought in online mode to maximize participation (Jan Bhagidari). Hon’ble Shiksha Mantri constituted three separate 
Independent National Jury comprising of eminent persons for selection of teachers.""",
                                                     ["image1", "image2"])
    descriptiveContent = descriptiveContentGenerator.generate_descriptive_content(dummyOriginalPressRelease)

    print("Generated Content:")
    print("Descriptive Content:")
    print(descriptiveContent.prId)

    print(descriptiveContent.to_json())
