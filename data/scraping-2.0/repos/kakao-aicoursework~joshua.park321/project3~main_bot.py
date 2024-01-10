import logging
from pprint import pformat

from project3.assets import AssetHelper
from project3.google_search import GoogleSearchHelper
from project3.langchain_helper import LangchainHelper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-16s %(levelname)-8s %(message)s ",
    datefmt="%Y-%m-%d %H:%M:%S"
)

TEMPLATE_PATH_BASE = 'prompt_templates'


def init_bot(template_path_base=TEMPLATE_PATH_BASE, path_db=None):
    langchain = LangchainHelper()
    asset_helper = AssetHelper(path_db=path_db)
    google_search_helper = GoogleSearchHelper()
    langchain.system_prompt = open(template_path_base + '/system.txt').read()

    return langchain, asset_helper, google_search_helper


def main_bot_loop(user_message, langchain, asset_helper, google_search_helper, template_path_base=TEMPLATE_PATH_BASE):
    chain_fallback = langchain.create_chain(template_file_path=template_path_base + '/fallback.txt', output_key='answer')
    chain_select_topic = langchain.create_chain(template_file_path=template_path_base + '/select_topic.txt', output_key='topic')
    chain_search_query = langchain.create_chain(template_file_path=template_path_base + '/generate_search_query.txt', output_key='search_query')
    chain_describe_topic = langchain.create_chain(
        template_file_path=template_path_base + '/describe_topic.txt',
        output_key='answer'
    )

    context = dict(
        topic_list=['카카오소셜', '카카오싱크', '카카오톡채널', 'NONE'],
        user_message=user_message,
    )

    context['topic'] = chain_select_topic.run(context)
    if context['topic'] == 'NONE':
        context['answer'] = chain_fallback.run(context)
        logging.info(pformat(context))
        return context['answer']

    context['search_query'] = chain_search_query.run(context)
    context['web_search_result'] = google_search_helper.search(context['search_query'])
    context['document'] = asset_helper.query(context['search_query'])
    context['answer'] = chain_describe_topic.run(context)

    logging.info(pformat(context))
    return context['answer']


if __name__ == "__main__":
    from init_keys import init_keys
    init_keys()
    langchain, asset_helper, google_search_helper = init_bot()
    asset_helper.load()
    while True:
        user_message = input('user: ')
        answer = main_bot_loop(user_message, langchain, asset_helper, google_search_helper)
        print(f'bot: {answer}')


