import os
import logging
from argparse import ArgumentParser
from datetime import datetime, timezone, timedelta

from typing import List, Dict

from .e5 import e5_embed, e5_embed_text
from .utils import compare_clusterings, cluster_louvain, cluster_print
from .. import CommonArguments
from ..esdl import Elastika
from ..esdl.article import Article
from ..oai.embed import openai_embed

logger = logging.getLogger('play.cluster')

cmap = {
    'ccfe00b9-d397-4e85-8310-1a2278ecb73f': 'PS',
    'a65c7372-9fbe-410c-93d7-4613d26488e7': 'DZ',
    '9fb98b28-6e82-4e30-8d36-7e3e9e09a9c0': 'NB',
    '7fd935a6-a1f5-42d1-8b5f-048dd54c07d1': 'NG',
    '011afa08-1b10-48d4-b0ea-cc05d8f7e2a9': 'CD'
}


def add_args(module_name: str, parser: ArgumentParser) -> None:
    # CommonArguments.result_dir(module_name, parser, ('-o', '--result_dir'))
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
        '-c', '--country', help='Articles selection country.', type=str
    )
    parser.add_argument(
        '-f', '--fields', help='Fields to embed.', type=str, default='tb', required=False,
        choices=['b', 'tb']
    )
    parser.add_argument(
        '-u', '--customer', help='Articles selection customer.', type=str,
        default='a65c7372-9fbe-410c-93d7-4613d26488e7'
    )
    parser.add_argument(
        '-l', '--e5_large', help='Enable large E5.', action='store_true', default=False
    )


def _get_articles(arg):
    requests = Elastika()
    requests.limit(1000)
    requests.filter_customer(arg.customer)
    if arg.country is not None:
        requests.filter_country(arg.country)
    # requests.field('vector_768___textonic_v1')

    articles: List[Article] = requests.gets(arg.start_date, arg.end_date)
    return articles


def cluster_compare(arg) -> int:
    a_dir = os.path.join(arg.tmp_dir, 'cluster_articles')
    if not os.path.exists(a_dir):
        os.makedirs(a_dir)

    articles: List[Article] = _get_articles(arg)
    openai_embed(articles, 'oai_ada_002', a_dir, arg.fields)
    e5_embed(articles, 'e5', arg.tmp_dir, arg.fields, arg.e5_large)

    if arg.customer in cmap.keys():
        arg.customer = cmap[arg.customer]

    oai_l_clusters = cluster_louvain(articles, 'oai_ada_002', 0.92)
    e5_l_clusters = cluster_louvain(articles, 'e5', 0.91)
    append = ''
    if arg.e5_large:
        append = '_large'
    f_prefix = arg.customer + append + '_' + arg.fields + '_' + arg.start_date + '_' + arg.end_date
    print('')
    print('========================== OpenAI ========================== ')
    cluster_print(oai_l_clusters, os.path.join(arg.tmp_dir, 'OpenAI-' + f_prefix + '.txt'))

    print('')
    print('==========================   E5   ========================== ')
    cluster_print(e5_l_clusters, os.path.join(arg.tmp_dir, 'E5-' + f_prefix + '.txt'))
    return 0


def cluster_test(arg) -> int:
    text = '''
    Vremenska napoved. 
    Popoldne bo deloma sončno, nastale bodo krajevne plohe in nevihte. V nedeljo bo  pretežno jasno.
    '''
    embeddings = e5_embed_text(arg.tmp_dir, text)
    print(embeddings)
    return 0
