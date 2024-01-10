import openai
from django.conf import settings
from edx_rest_api_client import client as rest_client

from flashcards.apps.cards.prompt import anki_prompt

openai.api_key = settings.OPENAI_API_KEY


def get_client(oauth_base_url=settings.LMS_ROOT_URL):
    """
    Returns an authenticated edX REST API client.
    """
    client = rest_client.OAuthAPIClient(
        oauth_base_url,
        settings.BACKEND_SERVICE_EDX_OAUTH2_KEY,
        settings.BACKEND_SERVICE_EDX_OAUTH2_SECRET,
    )
    client._ensure_authentication()  # pylint: disable=protected-access
    if not client.auth.token:  # pragma: no cover
        raise Exception('No Auth Token')  # pylint: disable=broad-exception-raised
    return client


def block_content_for_cards(course_id, block_id, source='lms'):
    block_id = block_id.replace(":", "$:")
    lms_url = f'{settings.LMS_ROOT_URL}/courses/{course_id}/xblock/aside-usage-v2:{block_id}::extractor_aside/handler/extract_handler'  # noqa
    # currently sourcing via the CMS doesn't work by auth and if auth is disabled walking
    # the handler in the CMS context poisons the cache and obliterates those blocks in the course
    # cms_url = f'{settings.CMS_ROOT_URL}/xblock/aside-usage-v2:{block_id}::extractor_aside/handler/extract_handler'

    handler_url = lms_url

    client = get_client()
    response = client.get(handler_url)

    if response.status_code < 200 or response.status_code > 300:
        response.raise_for_status()

    content = response.json()
    joined_content = '\n\n'.join(content['content'])
    return joined_content


def cards_from_openai(content):
    messages = [
        {"role": "system", "content": anki_prompt},
        {"role": "system", "content": content},
    ]

    result = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=1.0,
    )

    maybe_csv = result['choices'][0]['message']['content']
    return maybe_csv


def cards_from_block_id(course_id, block_id):
    content = block_content_for_cards(course_id, block_id)
    cards = cards_from_openai(content)
    return cards
