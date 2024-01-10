from argparse import ArgumentParser, Namespace
from dataclasses import asdict
import json
import logging
from autosub.cli.translation import OpenAITranslationCommand
from autosub.models.translation_context import TranslationContext
from autosub.utils import EnhancedJSONEncoder
from autosub.wiki.context_transformer.jp_creative_work_wikipage_transformer import JPCreativeWorkWikipageTransformer

from autosub.wiki.transport import WikiTransport


def create_context_parser(subparser: ArgumentParser):
    subparser.add_argument('wiki_domain')
    subparser.add_argument('wiki_title')
    openai_command = OpenAITranslationCommand()
    openai_command.configure_subparser(subparser)
    subparser.add_argument('output')


def generate_context(args: Namespace):
    logger = logging.getLogger(__name__)

    openai = OpenAITranslationCommand().create_client(args)
    transport = WikiTransport(
        base_url=args.wiki_domain,
        user_agent='Test',
    )

    logger.info("Querying wikipedia")
    wikipage = transport.retrieve_wikipage(args.wiki_title)
    context_transformer = JPCreativeWorkWikipageTransformer(
        llm_transport=openai,
        wikipage=wikipage,
        wiki_transport=transport,
    )
    context = TranslationContext(
        synopsis=context_transformer.prepare_synopsis(),
        phrases=context_transformer.prepare_phrases(),
    )
    logger.info("Writing context to file")
    with open(args.output, 'w', encoding='utf8') as f:
        json.dump(
            asdict(context),
            f,
            cls=EnhancedJSONEncoder,
            indent=4,
            ensure_ascii=False,
        )
