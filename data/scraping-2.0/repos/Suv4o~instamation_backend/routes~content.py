from flask_restful import Resource
from transformers import pipeline
from langchain import PromptTemplate, OpenAI, LLMChain
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer, util
from werkzeug.exceptions import BadRequest, NotFound

from utils.decorators import requires_auth
from utils.enums import ErrorResponse, ImageToTextModels
from config.environments import OPENAI_API_KEY
from models import Assets


class ContentRoute(Resource):
    """Content route definition"""

    @requires_auth
    def get(self, image_uuid):
        try:
            result = get_image_content(image_uuid)
            return result

        except Exception as e:
            return {"success": False, "message": str(e)}, ErrorResponse.BAD_REQUEST.value


class ImageToTextModel:
    """Image to text model definition"""

    def __init__(self, model_name):
        self.captioner = pipeline("image-to-text", model=model_name, max_new_tokens=512)

    def predict(self, image_url):
        """Predict method"""
        return self.captioner(image_url)


class LangChainLLMChainOpenAi:
    """LangChain LLMChain OpenAi definition"""

    def __init__(self, template, temperature, variables):
        self.variables = variables
        self.input_variables = list(variables.keys())
        self.prompt = PromptTemplate(template=template, input_variables=self.input_variables)
        self.llm_chain = LLMChain(
            prompt=self.prompt, llm=OpenAI(temperature=temperature, openai_api_key=OPENAI_API_KEY)
        )

    def predict(self):
        """Predict method"""
        output = self.llm_chain.predict(**self.variables)
        return output


def get_image_capture(model_name, image_url):
    """Get image capture method"""

    captioner = ImageToTextModel(model_name)
    image_capture = captioner.predict(image_url)
    return image_capture.pop().get("generated_text")


def get_image_classes(image_capture_salesforce, image_capture_microsoft, image_capture_nlpconnect):
    template = """Create a single list of all the sentence phrases, verbs and nouns from the following three sentences that describe the image:
                1. {image_capture_1}
                2. {image_capture_2}
                3. {image_capture_3}
                The sentence phrases such as: "sun set", "long exposure", "beautiful scenery", "nice view" etc. must not be separated into individual words. Instead, they must be kept as a single phrase.
                The output must be a single list that will only meaningful phrases, verbs and nouns that will be separated by commas.
                The output must not contain any geographical locations, names of people, names of cities or names of countries.
                """

    image_classes = LangChainLLMChainOpenAi(
        template=template,
        temperature=0,
        variables={
            "image_capture_1": image_capture_salesforce,
            "image_capture_2": image_capture_microsoft,
            "image_capture_3": image_capture_nlpconnect,
        },
    ).predict()

    image_classes = image_classes.replace("\n", "")
    image_classes = image_classes.split(",")
    image_classes = [word.strip() for word in image_classes]

    model_sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
    paraphrases = util.paraphrase_mining(model_sentence_transformer, image_classes)

    paraphrases_scores = []

    for paraphrase in paraphrases:
        score, i, j = paraphrase
        paraphrases_scores.append(
            {
                "score": score,
                "word_1": image_classes[i],
                "word_2": image_classes[j],
            }
        )

    paraphrases_scores = [paraphrase for paraphrase in paraphrases_scores if paraphrase["score"] >= 0.5]

    for paraphrase in paraphrases_scores:
        image_classes.append(paraphrase["word_1"] + " " + paraphrase["word_2"])

    return image_classes


def get_zero_shot_image_classes_top_scores(image_url, image_classes):
    """Get zero shot image classes scores method"""

    model_zero_shot_classification = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(text=image_classes, images=image, return_tensors="pt", padding=True)
    outputs = model_zero_shot_classification(**inputs)
    logits_per_image = outputs.logits_per_image
    scores_per_image = logits_per_image.tolist()[0]

    image_classes_with_scores = []

    for score in scores_per_image:
        if score >= 15:
            image_classes_with_scores.append({"score": score, "word": image_classes[scores_per_image.index(score)]})

    image_classes_with_scores = sorted(image_classes_with_scores, key=lambda k: k["score"], reverse=True)
    image_classes_string = (label["word"] for label in image_classes_with_scores)
    image_classes_string = ", ".join(image_classes_string)

    return image_classes_string


def get_image_description(image_classes_string):
    template = """Generate a description for an image that will be posted on instagram. The image has the following labels: {image_classes}
    Use the provided labels to write an instagram description of the image: 
    Step 1: Carefully analyze all labels to summarize the content of the image. Keep in mind that some labels may not accurately represent the true image content. Try your best to determine the topic of the image before proceeding with writing your description.
    Step 2: Write a description that can be used in an Instagram post for the users account. The caption describes the image and captures the essence of the moment. Also, remember that the image was taken from users camera, so you might need to act as a fist person when is needed about the story behind the image or how you captured the moment. This will help the audience connect with the image and understand its significance.
    Step 3: Generate hashtags that are relevant to the description and image content. Consider using hashtags that relate to the image, and use engaging and descriptive language. Also, try to generate as many hashtags as possible related to tourist attractions, or parts of the image shown. Hashtags are very important to engage with the audience. You must not use city names or country names in the hashtags. For example, instead of saying "#torontolife", you should say "#citylife". This is because we do not know where the image was captured.
    You must follow the following rules:
    1. Summarize the capture in a single sentence and ensure that the description and hashtags do not exceed the 2200-character limit. This limit is hardcoded for Instagram captions.  
    2. Do not use using time-related words such as "today", "yesterday", "last year", etc., since we do not know when the image was captured. 
    3. Do not use using words such as "Description:" or "Hashtags:" that explicitly indicate the start of the description or hashtags.
    4. Must not use city names or country names in the description. Instead, use general words such as "city", "country", "place", etc. For example, instead of saying "I visited Toronto", you should say "I visited a city". This is because we do not know where the image was captured.
    5. The image description should be descriptive and not contain wording such as "The image is most likely to be a mountain …". Instead, it should be something like "Mountain view on a nice summer day with a reflection in the lake …". Use your own imagination to come up with a nice caption. The three dots '...' in the examples indicate that the text should continue.
    6. It is good to include a personal touch in your writing. For example, you could say "This is an image I took..." or "This scenery was captured by me..." or "I had the opportunity to take a photo of this great view that I visited..."
    """

    result = LangChainLLMChainOpenAi(
        template=template,
        temperature=0.7,
        variables={
            "image_classes": image_classes_string,
        },
    ).predict()

    result = result.replace("\n", "")
    result_start = result.index("Description:") + 12 if "Description:" in result else 0
    result = result[result_start:].strip()

    return result


def get_image_url_from_db(image_uuid):
    try:
        asset = Assets.query.filter(Assets.aid == image_uuid).first()

        if not asset:
            raise NotFound("Image not found.")
        else:
            return asset.url

    except Exception as e:
        raise BadRequest(e)


def get_image_content(image_uuid):
    image_url = get_image_url_from_db(image_uuid)

    image_capture_salesforce = get_image_capture(ImageToTextModels.SALESFORCE.value, image_url)
    image_capture_microsoft = get_image_capture(ImageToTextModels.MICROSOFT.value, image_url)
    image_capture_nlpconnect = get_image_capture(ImageToTextModels.NLPCONNECT.value, image_url)

    image_classes = get_image_classes(image_capture_salesforce, image_capture_microsoft, image_capture_nlpconnect)

    image_classes_string = get_zero_shot_image_classes_top_scores(image_url, image_classes)

    result = get_image_description(image_classes_string)

    return result
