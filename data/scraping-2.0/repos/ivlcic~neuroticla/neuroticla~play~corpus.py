import os
import logging
import json
import csv
import pandas as pd
import regex
import openai
import numpy as np
import torch.nn.functional as functional

from argparse import ArgumentParser
from datetime import datetime, timezone, timedelta

from typing import List

from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from ..ner.prep.tokens import get_tokenizer
from .. import CommonArguments
from ..esdl import Elastika
from ..esdl.article import Article
from ..oai.constants import EMBEDDING_ENCODING, EMBEDDING_CTX_LENGTH
from ..oai.tokenize import truncate_text_tokens

logger = logging.getLogger('play.cluster')

__SOURCE_PATT = regex.compile(r'^[\p{Nd}\p{Lu}\s]+([,\s*])*(\s*(\d{1,2}\.\s*\d{1,2}\.\s*\d{4})([,\s*])*(\s*\p{Lu}{2,}[\p{Nd}\p{Lu}\s]+([,\s*])*(\s*\d{1,2}:\d{0,2})?)?)?\s*$')
__ACTOR_PATT = regex.compile(r'^[\p{Nd}\p{Lu}\s]+(\([\s\p{L}\p{Nd}\p{P}]+\))?\s*$')


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.result_dir('corpus', parser, ('-o', '--result_dir'))
    CommonArguments.tmp_dir(module_name, parser, ('-t', '--tmp_dir'))
    beginning_of_day = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
    parser.add_argument(
        '-s', '--start_date', help='Articles start selection date.', type=str,
        default=beginning_of_day.astimezone(timezone.utc).isoformat()
    )
    next_day = beginning_of_day + timedelta(days=1)
    parser.add_argument(
        '-e', '--end_date', help='Articles end selection date.', type=str,
        default=next_day.astimezone(timezone.utc).isoformat()
    )
    parser.add_argument(
        'customers', help='Article selection comma-separated customers or file name.', type=str
    )


def _average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def _filter_body(sentence_idx, sentence):
    if sentence_idx == 0:  # first sentence of a body section
        match = regex.match(__SOURCE_PATT, sentence)
    else:  # other sentences of a body section
        match = regex.match(__ACTOR_PATT, sentence)
    if match is None:
        return sentence
    else:
        return ''


def _e5_embed(arg, text):
    batch_dict = arg.tokenizer(
        ['passage: ' + text], max_length=512,
        padding=True, truncation=True, return_tensors='pt'
    )
    outputs = arg.model(**batch_dict)
    embeddings = _average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embeddings = functional.normalize(embeddings, p=2, dim=1)
    return embeddings.tolist()[0]


def _e5_token_compute(arg, text):
    tokens = arg.tokenizer(
        ['passage: ' + text], return_tensors='pt'
    )
    return len(tokens.encodings[0].tokens)


def _oai_embed(text):
    tokens = truncate_text_tokens(
        text,
        EMBEDDING_ENCODING,
        EMBEDDING_CTX_LENGTH
    )
    embedding = openai.embeddings.create(  # call OpenAI
        input=tokens, model="text-embedding-ada-002", timeout=10
    )
    return embedding.data[0].embedding  # extract vector from response


def _oai_token_compute(text):
    tokens = truncate_text_tokens(
        text,
        EMBEDDING_ENCODING,
        EMBEDDING_CTX_LENGTH
    )
    return len(tokens)


def _embed(arg, article: Article):
    arg.lang = article.language.split('-', 1)[0]
    if arg.lang == 'bs':  # since we don't have any tokenizer for Bosnian
        arg.lang = 'hr'
    if arg.lang == 'sq':  # since we don't have any tokenizer for Albanian
        arg.lang = 'en'

    tokenizer = get_tokenizer(arg)

    body = ''
    filtered = False
    mt = article.data['media']['type']['name']
    if mt == 'radio' or mt == 'tv':
        for line_idx, line in enumerate(article.body.split('\n')):
            tmp = _filter_body(line_idx, line)
            if tmp:
                body += '\n' if len(body) > 0 else ''
                body += tmp
            if not tmp and line:
                filtered = True
    if not body:
        body = article.body

    tdoc = tokenizer(article.title)
    bdoc = tokenizer(body)
    blen = len(body)

    bwt = 0
    for sentence in bdoc.sentences:
        bwt += len(sentence.tokens)
    twt = 0
    for sentence in tdoc.sentences:
        twt += len(sentence.tokens)

    b_spt = _e5_token_compute(arg, body)
    t_spt = _e5_token_compute(arg, article.title)

    b_oait = _oai_token_compute(body)
    t_oait = _oai_token_compute(article.title)

    article.data['body'] = {
        'text': article.body,
        'stats': {
            'chr': blen,
            'sent': len(bdoc.sentences),
            'w_t': bwt,
            'sp_t': b_spt,
            'oai_t': b_oait,
            'filter': True if filtered else False
        }
    }
    article.data['title'] = {
        'text': article.title,
        'stats': {
            'chr': len(article.title),
            'sent': len(tdoc.sentences),
            'w_t': twt,
            'sp_t': t_spt,
            'oai_t': t_oait,
        }
    }

    article.data['embed_e5'] = _e5_embed(arg, article.title + ' ' + body)
    article.data['embed_oai'] = _oai_embed(article.title + ' ' + body)

    article.data['stats'] = {
        'chr': article.data['title']['stats']['chr'] + article.data['body']['stats']['chr'],
        'sent': article.data['title']['stats']['sent'] + article.data['body']['stats']['sent'],
        'w_t': article.data['title']['stats']['w_t'] + article.data['body']['stats']['w_t'],
        'sp_t': article.data['title']['stats']['sp_t'] + article.data['body']['stats']['sp_t'],
        'oai_t': article.data['title']['stats']['oai_t'] + article.data['body']['stats']['oai_t']
    }


def _filter_write(arg, article: Article, data_path: str):
    article.data.pop('advertValue', None)
    article.data.pop('mediaReach', None)
    article.data.pop('translations', None)

    media = article.data.get('media')
    media.pop('publisher', None)
    media.pop('advertValue', None)
    media.pop('circulation', None)
    media.pop('providerId', None)
    media.pop('descriptor', None)
    media.pop('class', None)
    media.pop('language', None)
    tags = media.pop('tags', [])
    tags[:] = [tup for tup in tags if tup.get('class', '') == 'org.dropchop.jop.beans.tags.MediaType']
    media['type'] = {
        'name': tags[0]['name'],
        'uuid': tags[0]['uuid']
    }

    rubric = article.data.get('rubric')
    rubric.pop('advertValue', None)
    rubric.pop('providerId', None)
    rubric.pop('descriptor', None)
    rubric.pop('class', None)

    country = article.data.get('country')
    country.pop('tags', None)
    country.pop('descriptor', None)
    country.pop('class', None)

    for k, v in article.data.get('translations', {}).items():
        v.pop('bodyPages')
        v.pop('bodyOctetLen')
        v.pop('bodyCalculatedPages')
        v.pop('bodyMd5')
        v.pop('bodyBillingPages')
        v.pop('class')
        v.pop('uuid')

    tags = article.data.pop('tags', [])
    tags[:] = [tup for tup in tags if tup.get('class', '') != 'org.dropchop.jop.beans.tags.Genre']
    for i, t in enumerate(tags):
        t.pop('descriptor', None)
        t.pop('name', None)
        t.pop('_OOP', None)
        if t.pop('class', '') == 'org.dropchop.jop.beans.tags.CustomerTopic':
            t['type'] = 'topic'
        if t.pop('class', '') == 'org.dropchop.jop.beans.tags.CustomerTopicGroup':
            t['type'] = 'group'

    _embed(arg, article)
    logger.info("Done filtering and embedding [%s]", article)

    article.data['tags'] = tags
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    with open(os.path.join(data_path, article.uuid + '.json'), 'w', encoding='utf8') as json_file:
        json.dump(article.data, json_file, indent='  ', ensure_ascii=False)


def corpus_dump(arg) -> int:
    model_name = 'intfloat/multilingual-e5-base'
    arg.tokenizer = AutoTokenizer.from_pretrained(model_name)
    arg.model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=os.path.join(arg.tmp_dir, model_name)
    )

    start_date = datetime.fromisoformat(arg.start_date)
    end_date = datetime.fromisoformat(arg.end_date)
    if os.path.exists(arg.customers):
        with open(arg.customers) as f:
            customers = f.read().splitlines()
    else:
        customers = arg.customers.split(',')
    logger.info("Dumping [%s::%s] for %s", start_date, end_date, customers)
    for customer in customers:
        requests = Elastika()
        requests.limit(9999)
        requests.filter_customer(customer)
        requests.field(['rubric', 'url'])

        current_date = end_date
        while current_date > start_date:
            prev_day = current_date - timedelta(days=1)
            day_dir = os.path.join(
                arg.result_dir, str(prev_day.year), f"{prev_day.month:02d}", f"{prev_day.day:02d}"
            )
            articles: List[Article] = requests.get(prev_day, current_date)
            for a in articles:
                if not os.path.exists(os.path.join(day_dir, a.uuid + '.json')):
                    _filter_write(arg, a, day_dir)
            logger.info("Dumped [%s::%s] for [%s]", prev_day, current_date, customer)
            current_date = prev_day
    return 0


def corpus_correct(arg) -> int:
    start_date = datetime.fromisoformat(arg.start_date)
    end_date = datetime.fromisoformat(arg.end_date)
    if os.path.exists(arg.customers):
        with open(arg.customers) as f:
            customers = f.read().splitlines()
    else:
        customers = arg.customers.split(',')
    logger.info("Dumping [%s::%s] for %s", start_date, end_date, customers)
    for customer in customers:
        requests = Elastika()
        requests.limit(9999)
        requests.filter_customer(customer)
        requests.field(['rubric', 'url'])

        current_date = end_date
        while current_date > start_date:
            prev_day = current_date - timedelta(days=1)
            day_dir = os.path.join(
                arg.result_dir, str(prev_day.year), f"{prev_day.month:02d}", f"{prev_day.day:02d}"
            )
            articles: List[Article] = requests.get(prev_day, current_date)
            for a in articles:
                article_file = os.path.join(day_dir, a.uuid + '.json')
                if not os.path.exists(article_file):
                    continue
                with open(article_file, encoding='utf-8') as json_file:
                    try:
                        saved_article = json.load(json_file)
                    except:
                        logger.error("Unable to load json file [%s] for [%s].", article_file, a)
                        os.remove(article_file)
                        return 1

                saved_article.pop('mediaReach', None)
                url = a.data.pop('url', None)
                if url:
                    saved_article['url'] = url

                saved_media = saved_article.get('media')
                media = a.data.get('media')
                if media and media.get('mediaReach', None):
                    saved_media['mediaReach'] = media.get('mediaReach')

                saved_rubric = saved_article.get('rubric')
                rubric = a.data.get('rubric')
                if rubric and rubric.get('mediaReach', None):
                    saved_rubric['mediaReach'] = rubric.get('mediaReach')

                with open(article_file, 'w', encoding='utf8') as json_file:
                    json.dump(saved_article, json_file, indent='  ', ensure_ascii=False)

                logger.info("Corrected [%s]", a)
            logger.info("Corrected [%s::%s] for [%s]", prev_day, current_date, customer)
            current_date = prev_day
    return 0


def _calculate_percentiles(data, percentiles=None):
    if percentiles is None:
        percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    return {metric: np.percentile(data[metric], percentiles) for metric in data}


def corpus_stats_collect(arg) -> int:
    start_date = datetime.fromisoformat(arg.start_date)
    end_date = datetime.fromisoformat(arg.end_date)

    data = {
        'uuid': [],
        'published': [],
        'characters': [],
        'sentences': [],
        'word_tok': [],
        'sp_tok': [],
        'cl100k_tok': [],
        'country': [],
        'language': [],
        'media_type': [],
        'industries': [],
        'rel_path': [],
        'title': []
    }
    industry_map = {}
    map_file_name = os.path.join(arg.result_dir, 'tag_map.csv')
    with open(map_file_name, encoding='utf-8') as map_file:
        try:
            reader = csv.reader(map_file)
            for row in reader:
                # Assuming the first column is the key and the second is the value
                key = row[0]
                value = row[1]
                industry_map[key] = value
        except:
            logger.error("Unable to load CSV tag map file [%s].", map_file_name)
            return 1
    distinct_industries = set(industry_map.values())
    print(distinct_industries)

    current_date = end_date
    while current_date > start_date:
        prev_day = current_date - timedelta(days=1)
        rel_path = os.path.join(
            str(prev_day.year), f"{prev_day.month:02d}", f"{prev_day.day:02d}"
        )
        day_dir = os.path.join(arg.result_dir, rel_path)
        file_names = os.listdir(day_dir)
        num_files = 0
        for article_file in file_names:
            # check if the file is a JSON file
            if not article_file.endswith('.json'):
                continue
            article_file = os.path.join(day_dir, article_file)
            if not os.path.exists(article_file):
                continue
            with open(article_file, encoding='utf-8') as json_file:
                try:
                    saved_article = json.load(json_file)
                except:
                    logger.error("Unable to load json file [%s].", article_file)
                    os.remove(article_file)
                    return 1
                data['uuid'].append(saved_article['uuid'])
                data['published'].append(saved_article['published'])
                data['characters'].append(saved_article['stats']['chr'])
                data['sentences'].append(saved_article['stats']['sent'])
                data['word_tok'].append(saved_article['stats']['w_t'])
                data['sp_tok'].append(saved_article['stats']['sp_t'])
                data['cl100k_tok'].append(saved_article['stats']['oai_t'])
                data['country'].append(saved_article['country']['name'])
                data['language'].append(saved_article['language'].split('-', 1)[0])
                data['media_type'].append(saved_article['media']['type']['name'])

                ids = set([industry_map[t['uuid']] for t in saved_article['tags'] if t['uuid'] in industry_map])
                data['industries'].append(ids)
                data['rel_path'].append(rel_path)
                data['title'].append(saved_article['title']['text'])
                num_files += 1
        logger.info("Collected [%s = %s].", prev_day, num_files)
        current_date = prev_day

    # Calculate percentiles
    # percentiles = _calculate_percentiles(data)
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(arg.result_dir, f'stats-{arg.start_date}_{arg.end_date}.csv'))
    print(df)
    return 0
