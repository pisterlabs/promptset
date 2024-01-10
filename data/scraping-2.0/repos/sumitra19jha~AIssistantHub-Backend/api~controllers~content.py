from http import HTTPStatus
import math
import nltk
import spacy
import openai
import textstat
import language_tool_python
from typing import List

import gensim
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from nltk.stem import WordNetLemmatizer
from scipy.spatial.distance import cosine
from api.utils.classifier_models import ClassifierModels
from api.utils.input_preprocessor import InputPreprocessor
from api.utils.instructor import Instructor
from api.utils.prompt import PromptGenerator
from api.utils.scrapper import AssistantHubScrapper
from api.utils.socket import Socket
from api.utils.validator import APIInputValidator

from api.models import db
from config import Config
from api.assets import constants
from api.utils.request import bad_response, response
from api.models.content import Content
from api.middleware.error_handlers import internal_error_handler
from api.controllers import dashboard as dashboard_controller


openai.api_key = Config.OPENAI_API_KEY
nlp = spacy.load("en_core_web_md")
tool = language_tool_python.LanguageTool("en-US");
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def best_content(prompt: str, generated_contents: List[str]) -> str:
    scores = []
    index = 0
    for content in generated_contents:
        relevance_score = relevance(prompt, content)
        coherence_score = coherence(content)
        readability_score = readability(content)
        grammar_score = grammar_and_spelling(content)
        length_score = length(content)
        index = index + 1

        # Assign weights to each metric and calculate the overall score
        total_score = (
            0.4 * relevance_score
            + 0.2 * coherence_score
            + 0.2 * readability_score
            + 0.1 * grammar_score
            + 0.1 * length_score
        )

        scores.append(total_score)

    best_index = scores.index(max(scores))
    return generated_contents[best_index]

# Measure the relevance of User Message with the Update generated
def relevance(prompt: str, content: str) -> float:
    # Generate embeddings for the prompt and the generated content
    embeddings = embed([prompt, content])

    # Calculate the cosine similarity between the embeddings
    similarity = 1 - cosine(embeddings[0], embeddings[1])

    return similarity

def coherence(content: str) -> float:
    # Tokenize and preprocess the content
    tokens = preprocess(content)

    # Create a dictionary representation of the documents
    id2word = corpora.Dictionary([tokens])

    # Create a Bag of Words representation of the content
    corpus = [id2word.doc2bow(text) for text in [tokens]]

    # Create an LDA model using Gensim
    lda_model = gensim.models.LdaMulticore(corpus, num_topics=1, id2word=id2word, passes=10, workers=2)

    # Calculate coherence score using the CoherenceModel from Gensim
    coherence_model_lda = CoherenceModel(model=lda_model, texts=[tokens], dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()

    return coherence_lda

def preprocess(text: str) -> List[str]:
    lemmatizer = WordNetLemmatizer()

    def lemmatize_stemming(token):
        return lemmatizer.lemmatize(token, pos='v')

    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


def readability(content: str) -> float:
    flesch_reading_ease = textstat.flesch_reading_ease(content)
    return flesch_reading_ease


def grammar_and_spelling(content: str) -> float:
    matches = tool.check(content)
    grammar_score = 1 - len(matches) / len(content)
    return grammar_score


def length(content: str) -> float:
    # Calculate the length score based on your preferred content length
    preferred_length = 500
    length_score = 1 - abs(len(content) - preferred_length) / preferred_length
    return length_score

def format_name(name: str) -> str:
    formatted_name = name.replace(" ", "_")
    return formatted_name[:64]


@internal_error_handler
def create(user, topic, platform, keywords, length, urls, user_ip):
    is_allowed, purchase = dashboard_controller.is_user_have_sufficient_points(user.id)

    if not is_allowed:
        return bad_response(
            message="You don't have enough points to create a project.",
            status_code=HTTPStatus.BAD_REQUEST,
        )

    # Validate the user input
    validation_response = APIInputValidator.validate_content_input_for_social_media(
        topic,
        platform,
        length,
    )

    if validation_response:
        return validation_response

    try:
        # Preprocess the user input
        processed_input = InputPreprocessor.preprocess_user_input_for_social_media_post(
            topic,
            urls,
            length,
            platform
        )
    except ValueError as e:
        return response(
            success=False,
            message=str(e),
            status_code=HTTPStatus.BAD_REQUEST,
        )
    except Exception as e:
        return response(
            success=False,
            message=str(e),
            status_code=HTTPStatus.BAD_REQUEST,
        )

    url_map = {}
    for index in range(len(processed_input['urls'])):
        url_map[index] = processed_input['urls'][index]

    content_data = Content(
        user_id=user.id,
        type="SOCIAL_MEDIA_POST",
        topic=processed_input['topic'],
        keywords=keywords,
        length=processed_input['length'],
        model=Config.OPENAI_MODEL,
        platform=processed_input['platform'],
        urls=url_map,
    )
    db.session.add(content_data)
    db.session.flush()

    Instructor.initiate_chat_instruction_for_social_media(
        user.name,
        content_data,
    )

    # Call the Node.js server to create a room
    Socket.create_room_for_content(
        content_data.id,
        content_data.user_id,
    )

    return response(
        success=True,
        message=constants.SuccessMessage.content_generated,
        contentId=content_data.id,
    )

@internal_error_handler
def fetch_content(user, content_id, user_ip):
    points = 0.0
    is_allowed, purchase = dashboard_controller.is_user_have_sufficient_points(user.id)

    if not is_allowed:
        return bad_response(
            message="You don't have enough points to create a content.",
        )

    if content_id is None:
        return bad_response(
            message="Content Id is required.",
        )
    
    content_data = Content.query.filter(
        Content.id == content_id,
        Content.user_id == user.id,
    ).first()

    if content_data.system_message is not None:
        return response(
            success=True,
            message=constants.SuccessMessage.content_generated,
            content=content_data.content_data,
        )

    is_opinion, total_tokens = ClassifierModels.is_the_topic_opinion_based(
        content_data.topic,
    )

    # Cost of Classification
    points = points + ((total_tokens * 0.02)/1000)

    if is_opinion:
        web_searched_results, total_point = AssistantHubScrapper.search_and_crawl(
            content_data.topic,
            user_ip,
        )

        # Cost of Crawl
        points = points + total_point

        web_content = ""
        for result in web_searched_results:
            web_content = web_content + result['website'] + "\n\n"
            web_content = web_content + result['content'] + "\n\n"

        system_message, user_message = PromptGenerator.generate_messages_on_opinion_for_social_media(
            content_data.topic,
            content_data.platform,
            content_data.length,
            web_content
        )
    else:
        web_searched_results = None
        system_message, user_message = PromptGenerator.generate_messages_for_social_media(
            content_data.topic,
            content_data.platform,
            content_data.length,
        )

    content_data.system_message = system_message
    content_data.user_message = user_message
    purchase.points = purchase.points - math.ceil((points * 100))
    
    db.session.commit()
    return response(
        success=True,
        message=constants.SuccessMessage.content_generated,
        content=content_data.content_data,
    )


@internal_error_handler
def update_content(user_id, user_name, content_id, message):
    content_data = (
        Content.query.filter(
            Content.id == content_id,
            Content.user_id == user_id,
            Content.status == constants.ContentStatus.SUCCESS,
        ).first()
    )

    system_message = {
        "role": "system",
        "content": "You are a highly specialised content creator GPT. We have generated content for user and then interacted with user to understand his needs. User has provided the feedback on changes that needs to be done in content. Your job is to provide a new content.",
        "name": "AIsstant_Hub",
    }

    user_message = {
        "role": "user",
        "content": f"{content_data.model_response}\n\n\"{message}\"",
        "name": format_name(user_name),
    }

    assistant_response = openai.ChatCompletion.create(
        model=Config.OPENAI_MODEL,
        messages=[system_message, user_message],
        temperature=0.7,
        n=5,
        presence_penalty=0,
        user=str(user_id),
        frequency_penalty=0,
    )

    contents_by_model = [resp_data["message"]["content"] for resp_data in assistant_response["choices"]]
    best_content_data = best_content(message, contents_by_model)

    return response(
        success=True,
        message="Content generated.",
        content=best_content_data,
    )
