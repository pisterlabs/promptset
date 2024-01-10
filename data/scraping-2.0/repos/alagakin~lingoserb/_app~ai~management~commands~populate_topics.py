import os

import openai
from django.core.management.base import BaseCommand, CommandError
import json

from learn_serbian.utils import transliterate
from topics.models import Topic
from words.models import Word, Translation
import logging

logger = logging.getLogger('openai')


class Command(BaseCommand):
    help = "Imports base 3000 words with English and Russian translations"
    prompt = """Provide 30 Serbian words for topic '%s' with translation
        to Russian and English. Use singular for of nouns and infinitives for
        verbs. Use JSON only, like this {
    "%s": [
        {
            "serbian": "српска реч",
            "russian": "русское слово",
            "english": "english word"
        }, ]"""

    def handle(self, *args, **options):
        openai.api_key = os.getenv('OPENAI_KEY')
        topic = Topic.objects.filter(parent_id__gt=0).order_by(
            '-words').first()
        topic_en = topic.translations.filter(lang='en').first().title
        prompt = self.prompt % (topic_en, topic_en)

        chat_completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=[
                {"role": "user",
                 "content": prompt}]
        )
        result = chat_completion['choices'][0]['message']['content']
        logger.info(result)
        result = json.loads(result)
        pairs = result[topic_en]
        logger.info(pairs)

        for pair in pairs:
            sr_word = transliterate(pair['serbian'].lower())
            ru_word = pair['russian'].lower()
            en_word = pair['english'].lower()

            word = Word.objects.get_or_create(title=sr_word)
            word_instance = word[0]

            Translation.objects.get_or_create(lang='ru',
                                              word=word_instance,
                                              title=ru_word)
            Translation.objects.get_or_create(lang='en',
                                              word=word_instance,
                                              title=en_word)
            word_instance.topics.add(topic)

