import datetime
import io

import pytest
from openai._base_client import HttpxBinaryResponseContent

from autobots.conn.aws.aws_s3 import get_public_s3
from autobots.conn.openai.openai_client import get_openai
from autobots.conn.openai.openai_audio.speech_model import SpeechReq
from autobots.conn.openai.openai_audio.translation_model import TranslationReq
from autobots.core.utils import gen_hash


@pytest.mark.asyncio
async def test_translation_happy_path(set_test_settings):
    # # create audio file
    openai_client = get_openai()
    input = f"Namaste, main AutobotX hoon. main aapaki sahaayata ke lie yahaan hoon. Aaj ki tareekh {datetime.datetime.now().strftime('%B %d, %Y')} hai."
    params = SpeechReq(input=input)
    resp:  HttpxBinaryResponseContent = await openai_client.openai_audio.speech(speech_req=params)
    assert resp is not None
    url = await get_public_s3().put_file_obj(io.BytesIO(resp.content),f"{gen_hash(input)}.{params.response_format}")
    assert url != ""

    # Create Translation
    translation_req = TranslationReq(
        file_url=str(url)
    )
    translation = await openai_client.openai_audio.translation(translation_req)

    assert translation is not None
    assert "i am here to help you" in translation.text.lower()

