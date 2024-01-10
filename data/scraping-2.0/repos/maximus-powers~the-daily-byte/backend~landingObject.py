from memeObject import MemeObject
from imageObject import ImageObject
from dbObject import dbObject
from llmObject import LLMObject
import openai

class landingObject:
    def __init__(self):
        self.meme_obj = MemeObject()
        self.img_obj = ImageObject()
        self.db_obj = dbObject()
        self.llm_obj = LLMObject()

    def gen_landing_content(self, user_id):
        categories = self.db_obj.get_user_categories(user_id).split(',')    
        all_headlines_urls = {}
        for category in categories:
            cat_content = self.db_obj.call_category_content(user_id, category)
            for item in cat_content:
                all_headlines_urls[item['headline']] = item['url']

        ranked_dict = self.llm_obj.rank_dictionary(all_headlines_urls)

        for article in ranked_dict:
            headline_temp = self.llm_obj.summarize_article(article['url'], 25)
            summary_sentence_temp = self.llm_obj.summarize_article(article['url'], 75)

            if headline_temp is not False and summary_sentence_temp is not False:
                top_url = article['url']

                # jesus llms are so inconsistent
                if headline_temp.endswith('.'): top_headline = headline_temp[:-1]
                else: top_headline = headline_temp

                if summary_sentence_temp.endswith('.'): summary_sentence = summary_sentence_temp[:-1]
                else: summary_sentence = summary_sentence_temp

                break

        # init vars for the loop
        max_attempts = 3 
        attempt = 0
        image_url = None
        image_blob = None

        while attempt < max_attempts and image_url is None:
            try:
                # get prompt for image
                prompt = self.llm_obj.gen_image_prompt(top_headline, summary_sentence)  # string --> string

                # try to generate image
                image_url = self.img_obj.generate_image(prompt)  # string --> string url
                image_blob = self.img_obj.download_image_as_blob(image_url)  # string url --> binary blob

            except openai.BadRequestError as e:
                if 'content_policy_violation' in str(e):
                    attempt += 1
                    print(f"Attempt {attempt}: Content policy violation, trying with a safer prompt.")

                    # modify prompt to be safer
                    prompt = self.llm_obj.gen_safer_image_prompt(prompt)
                else:
                    raise  # re-raise the exception if it's not a content policy violation

        if image_url is None:
            print("Failed to generate an image after several attempts.")

        # # generate meme term
        # meme_term = self.llm_obj.gen_meme_term(top_headline) # string --> string

        # # generate meme url
        # meme_url = self.meme_obj.find_meme(meme_term) # string --> string

        return {
            'headline': top_headline,
            'summary': summary_sentence,
            'url': top_url,
            'alt_text': 'not working rn',
            'image_blob': image_blob
        }


